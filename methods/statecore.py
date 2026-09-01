"""StateCore as a memory method for MemoryAgentBench.

StateCore (github.com/yul761/StateCore) is an auditable memory engine: writes go through a
deterministic pipeline, a corrected fact supersedes its predecessor on a recorded chain
(`supersededBy`), and retirement/discards are logged rather than silent. It is driven here
through its published MCP front end -- the wrapper spawns

    npx -y statecore-mcp@0.6.0 --data <fresh temp dir>

and speaks JSON-RPC over stdio (newline-delimited, per the MCP stdio transport). Nothing is
installed into this venv and no service needs starting by hand; the only requirement is Node
>= 20 on PATH. The version is pinned so a run is reproducible from this file alone
(STATECORE_MCP_SPEC overrides, for testing a newer release without editing it).

TWO ARMS, like the knowl pair. `statecore_digest` in the agent config picks the arm, so a run
is reproducible from the config alone (STATECORE_DIGEST=1/0 overrides for one-off ablations):

  * default (deterministic): zero model calls on the memory side. Writes take the note path --
    a write that reads as a revision of an active fact supersedes it in place
    (short-token-preserving similarity, so "deadline is May 3" vs "deadline is May 4" replaces
    rather than accumulates) -- and retrieval is lexical (ASCII words + CJK bigrams) over facts
    and events. The only LLM in the loop is the shared reader every method uses; this row's
    score is the floor of the engine's LLM-assisted mode at zero memory-side token cost. The
    known blind spot is entity substitution ("prefers X" vs "prefers Y"): low lexical overlap,
    so both survive as active facts -- which is exactly what the digest arm is for.
  * digest (LLM-assisted): writes are stored as events and the engine's own distillation runs
    -- LLM extraction into supersession-tracked facts, semantic conflict resolution, entity
    vocabulary. The spawned process gets FEATURE_LLM=true and inherits OPENAI_API_KEY (the same
    key the harness already requires for the reader); the digest model is gpt-5-mini, the
    engine's recommended distillation model (its runtime sends reasoning_effort, which the
    gpt-4o family rejects, so the engine is operated with gpt-5-class models; STATECORE_MODEL_NAME
    overrides). Distillation runs at a pending-events threshold during ingestion; flush then
    restarts the process once (a startup catch-up pass digests the tail) and polls the `facts`
    tool until the distilled state is stable before the first query.

The delta between the two rows is the measured value of the engine's distillation -- that is
the point of shipping both, and it is stated here so nobody reads either row alone as the
system's spend-matched score.

NORMALIZED INPUT. Identical to the knowl/agentmemory rows: the parsed fact list, in context
order, one record per write, via `parse_fact_lines` (reused from methods.agentmemory). Same
records, no system-specific extraction. Facts over the note path's 500-char cap (rare in these
datasets) are stored as events instead -- still retrievable, just outside the supersession
machinery.

ISOLATION. Every context gets a fresh --data directory (its own SQLite file), so contexts
cannot see each other. The harness has no end-of-context hook, so cleanup is handoff-shaped:
initializing the client for context N closes context N-1's server and deletes its store, and
an atexit hook releases the last one -- at most one Node process and one tempdir are alive at
any time on a multi-context split.

READER. Mirrors `_handle_bm25_rag` via the shared knowl helpers, exactly like the agentmemory
row: same query extraction, "Memory i:" labels, same system template. Retrieval contents are
the active facts recall returns (already relevance-ranked and budget-packed by the engine),
then raw events, truncated to retrieve_num.
"""

import atexit
import json
import os
import shutil
import subprocess
import tempfile
import time

from methods.agentmemory import parse_fact_lines

DEFAULT_SPEC = "statecore-mcp@0.6.0"
PROTOCOL_VERSION = "2025-06-18"


class StateCoreMcpClient:
    """Minimal MCP-over-stdio client. Requests are sequential, so plain blocking reads on the
    child's stdout are enough; the first `npx -y` run downloads the package and generates its
    database client, which can take a minute -- later runs start in ~2s."""

    def __init__(self, data_dir, spec, extra_env=None):
        env = dict(os.environ)
        env.update(extra_env or {})
        self.proc = subprocess.Popen(
            ["npx", "-y", spec, "--data", data_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit: the server prints "[statecore-mcp] ready over stdio" there
            text=True,
            bufsize=1,
            env=env,
        )
        self._next_id = 0
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "memoryagentbench-statecore", "version": "1"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _request(self, method, params):
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("statecore-mcp exited before replying (is Node >= 20 on PATH?)")
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            if message.get("id") != request_id:
                continue  # notifications / unrelated traffic
            if "error" in message:
                raise RuntimeError(f"statecore-mcp error: {message['error']}")
            return message["result"]

    def call_tool(self, name, arguments):
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text = result["content"][0]["text"]
        if result.get("isError"):
            raise RuntimeError(f"statecore-mcp tool {name} failed: {text}")
        return json.loads(text)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class StateCoreClient:
    def __init__(self, spec, digest=False):
        self.chunks = []
        self.flushed = False
        self.facts = 0
        self.superseded = 0
        self.spec = spec
        self.digest = digest
        self.data_dir = tempfile.mkdtemp(prefix="mab_statecore_")
        # FEATURE_LLM gates the engine's own distillation; the key comes from the
        # environment the harness already has (the engine falls back to
        # OPENAI_API_KEY). The distillation model is the engine's recommended
        # gpt-5-mini -- its API is operated with gpt-5-class models.
        # MODEL_TIMEOUT_MS: the engine's default LLM timeout is 20s, tuned for
        # short interactive calls; a large-backlog distillation chunk on a
        # reasoning model routinely exceeds it and the run dies as an abort.
        self.extra_env = (
            {
                "FEATURE_LLM": "true",
                "MODEL_NAME": os.environ.get("STATECORE_MODEL_NAME", "gpt-5-mini"),
                "MODEL_TIMEOUT_MS": os.environ.get("STATECORE_MODEL_TIMEOUT_MS", "120000"),
            }
            if digest
            else {}
        )
        self.mcp = StateCoreMcpClient(self.data_dir, spec, self.extra_env)

    def close(self):
        """Terminate the server and delete the store. Idempotent."""
        if self.mcp is not None:
            self.mcp.close()
            self.mcp = None
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def add(self, text):
        self.chunks.append(text)

    def flush(self):
        """Write every parsed fact, in context order. Idempotent, like the agentmemory row.

        Order is the only recency signal the task provides; StateCore's revision matcher is what
        turns "later similar fact" into supersession rather than accumulation.
        """
        if self.flushed:
            return {"facts": self.facts, "superseded": self.superseded}

        facts = parse_fact_lines("".join(self.chunks))
        superseded = 0
        for fact in facts:
            if self.digest:
                # Event path: the engine's own distillation extracts and supersedes.
                # Threshold digests run in the background while ingestion continues.
                self.mcp.call_tool("remember", {"text": fact[:2000], "consolidate": True})
            elif len(fact) <= 500:
                result = self.mcp.call_tool("remember", {"text": fact})
                if result.get("superseded") is not None:
                    superseded += 1
            else:
                # Over the note cap: stored as an event -- retrievable, but outside supersession.
                self.mcp.call_tool("remember", {"text": fact[:2000], "consolidate": True})

        if self.digest:
            self._settle()

        self.facts = len(facts)
        self.superseded = superseded
        self.flushed = True
        arm = "digest" if self.digest else "note"
        print(f"\nstatecore flush ({arm}): {self.facts} facts, {superseded} superseded at write\n")
        return {"facts": self.facts, "superseded": superseded}

    def _settle(self, poll_seconds=5, stable_rounds=2, max_seconds=None, max_restarts=20):
        """Wait for the engine's background distillation to finish.

        Threshold digests fire during ingestion but leave a pending tail, and a
        single startup catch-up pass digests one batch per scope -- with a large
        backlog one pass is not the whole backlog. So: restart the process
        (reopening the store runs a catch-up pass), poll the `facts` tool until
        the distilled state stops changing for `stable_rounds` consecutive
        reads, and repeat until a restart no longer changes the stable state --
        that is the fixpoint where another pass has nothing left to digest. Time
        spent here is charged to memory_construction_time, where it belongs.
        """
        # A digest pass consumes a bounded batch (~40 events), so the number of
        # passes needed is a function of how many facts were written -- computed,
        # not guessed. The facts-snapshot fixpoint alone under-counts: on
        # template-heavy corpora a fresh pass's output can be entirely deduped
        # away, leaving the snapshot unchanged while a backlog remains (pending
        # counts are not observable over MCP today).
        if max_seconds is None:
            max_seconds = int(os.environ.get("STATECORE_SETTLE_SECONDS", "3600"))
        required_passes = (len(parse_fact_lines("".join(self.chunks))) // 40) + 2
        max_restarts = max(max_restarts, required_passes)
        deadline = time.time() + max_seconds
        settled_before_restart = None
        passes = 0
        for _ in range(max_restarts):
            passes += 1
            self.mcp.close()
            self.mcp = StateCoreMcpClient(self.data_dir, self.spec, self.extra_env)
            previous = None
            stable = 0
            while time.time() < deadline:
                snapshot = json.dumps(self.mcp.call_tool("facts", {}), sort_keys=True)
                if snapshot == previous and snapshot != "[]":
                    stable += 1
                    if stable >= stable_rounds:
                        break
                else:
                    stable = 0
                previous = snapshot
                time.sleep(poll_seconds)
            else:
                print("\nstatecore settle: hit max_seconds with distillation still moving; querying as-is\n")
                return
            if previous == settled_before_restart and passes >= required_passes:
                return  # enough passes for the backlog, and a fresh one changed nothing
            settled_before_restart = previous
        print("\nstatecore settle: hit max_restarts with distillation still moving; querying as-is\n")

    def query(self, text, k):
        result = self.mcp.call_tool("recall", {"query": text, "maxChars": 16000})
        facts = [f["content"] for f in (result.get("factRegistry") or []) if f.get("content")]
        events = [e["content"] for e in (result.get("events") or []) if e.get("content")]
        # Interleave the two layers rather than concatenating facts-first: each
        # layer is relevance-ranked by the engine, but distillation is selective
        # -- on a partially distilled store, facts-first let a handful of facts
        # crowd every event out of the top-k, and it was the event layer's
        # lexical index doing the heavy lifting. Interleaving keeps the top of
        # BOTH rankings inside the reader's window.
        contents = []
        if result.get("digest"):
            contents.append(result["digest"])
        for pair in range(max(len(facts), len(events))):
            if pair < len(facts):
                contents.append(facts[pair])
            if pair < len(events):
                contents.append(events[pair])
        return contents[:k]


# main.py builds a fresh AgentWrapper per context and never releases the old one, so each
# context would leak a Node process and a tempdir for the life of the run. Contexts are
# processed sequentially, so a single module-level slot is enough: the next context's init
# releases the previous context's client, and atexit releases the last.
_active_client = None


def _release_active_client():
    global _active_client
    if _active_client is not None:
        _active_client.close()
        _active_client = None


atexit.register(_release_active_client)


def initialize_statecore_agent(agent, agent_config=None):
    global _active_client
    config = agent_config or {}
    agent.retrieve_num = config["retrieve_num"]
    agent.context = ""
    agent.agent_start_time = time.time()

    spec = os.environ.get("STATECORE_MCP_SPEC", DEFAULT_SPEC)
    digest = bool(config.get("statecore_digest", False))
    override = os.environ.get("STATECORE_DIGEST")
    if override is not None:
        digest = override not in ("0", "false", "")
    _release_active_client()
    agent.statecore = StateCoreClient(spec, digest=digest)
    _active_client = agent.statecore
    arm = "digest" if digest else "note"
    print(f"\n\nstatecore ({spec}, {arm} arm) embedded at {agent.statecore.data_dir}\n\n")


def handle_statecore_agent(agent, message, memorizing, query_id, context_id):
    """Mirror `_handle_bm25_rag`: same query extraction, same reader assembly."""
    from methods.knowl import build_reader_messages, format_retrieval_memory_string
    from utils.templates import get_template

    if memorizing:
        agent.statecore.add(message)
        return "Memorized"

    start_time = time.time()
    stats = agent.statecore.flush()
    memory_construction_time = time.time() - start_time

    retrieval_query = agent._extract_retrieval_query(message)
    contents = agent.statecore.query(retrieval_query, agent.retrieve_num)
    retrieval_memory_string = format_retrieval_memory_string(contents)

    system_message = get_template(agent.sub_dataset, "system", agent.agent_name)
    format_message = build_reader_messages(retrieval_memory_string, message, system_message)

    response = agent._create_oai_client().chat.completions.create(
        model=agent.model,
        messages=format_message,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens if "gpt-4" in agent.model else None,
    )

    query_time_len = time.time() - start_time - memory_construction_time
    print(f"\nstatecore stats: {stats}\n")

    return agent._create_standard_response(
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        memory_construction_time,
        query_time_len,
    )
