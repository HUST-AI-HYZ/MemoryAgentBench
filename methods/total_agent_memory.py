"""total-agent-memory as a memory method for MemoryAgentBench.

total-agent-memory (github.com/vbcherepanov/total-agent-memory, "TAM") is a local memory server
for agents: SQLite + FTS5 + local embeddings, spoken to over MCP. It runs here as its own process
over MCP's stdio transport, so nothing is installed into this venv:

    uv tool install total-agent-memory==14.3.1       # or: pipx install total-agent-memory==14.3.1
    export TAM_COMMAND=total-agent-memory            # optional, this is the default

14.3.1 is the minimum: 14.3.0 deduplicated near-identical facts, so an update that changed one
value ("... is Argentina" -> "... is Armenia") was dropped and the old value kept.

The harness builds a new agent per context, and each agent starts its own TAM process on a fresh
store in a temporary directory (TAM_MEMORY_DIR), deleted at exit. The
TAM process receives no API keys: in its default fast mode it calls no LLM on save or recall, and
withholding the keys guarantees its background workers cannot either. Nothing leaves the machine.

WHAT IS MEASURED. Writes go through `memory_save`, reads through `memory_recall`, the search
path any client of TAM uses. TAM does not supersede facts at write time in this mode: a later
fact about the same subject is stored beside the earlier one, and both stay retrievable. What
TAM returns with every hit is the date the record was written (`created_at`). The two configs
differ only in whether that date reaches the reader:

  - `tam_show_recorded: true` prefixes each retrieved fact with its recorded time;
  - `tam_show_recorded: false` passes the fact text alone, like the other methods.

TAM's own latest-wins answering lives in `memory_answer`, which returns an answer rather than
passages; it is not used here, so the reader is the harness's gpt-4o-mini in both configs.

NORMALIZED INPUT. The same parsed fact list the knowl and agentmemory methods use
(`parse_fact_lines`), one `memory_save` per fact, in context order. Saves are sequential, so
recorded times increase strictly with the fact's position in the context.

FRAMING. MCP stdio is newline-delimited JSON-RPC. TAM writes its logs to stderr, and every
response is matched to its request id: a line on stdout that is not JSON, or a response to a
different id, raises instead of being skipped, so a desynchronised stream fails the run rather
than scoring it.
"""

import atexit
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time

DEFAULT_COMMAND = "total-agent-memory"
PROTOCOL_VERSION = "2025-06-18"
STARTUP_TIMEOUT_S = 300
CALL_TIMEOUT_S = 300
# Variables TAM needs to run; everything else, API keys included, is withheld.
PASSED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "HF_HOME", "FASTEMBED_CACHE_PATH")


class TamError(RuntimeError):
    pass


class TamProcess:
    """One TAM server over MCP stdio, with its own temporary store."""

    def __init__(self, command):
        self.memory_dir = tempfile.mkdtemp(prefix="mab-tam-")
        env = {name: os.environ[name] for name in PASSED_ENV if name in os.environ}
        env.update({
            "TAM_MEMORY_DIR": self.memory_dir,
            "MCP_TRANSPORT": "stdio",
            "MEMORY_MODE": "fast",
            "MEMORY_LLM_ENABLED": "false",
        })
        self.proc = subprocess.Popen(
            command.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.next_id = 0
        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "memoryagentbench", "version": "1"},
        }, STARTUP_TIMEOUT_S)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        atexit.register(self.close)

    def _send(self, message):
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _read(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TamError(f"no response from TAM within {timeout}s") from None
        if line is None:
            raise TamError(f"TAM exited with code {self.proc.wait()}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise TamError(f"non-protocol line on TAM stdout: {line[:200]!r}") from exc

    def _request(self, method, params, timeout):
        self.next_id += 1
        request_id = self.next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = self._read(timeout)
            if "id" not in message:
                continue  # server notification (log, progress)
            if message["id"] != request_id:
                raise TamError(f"response id {message['id']} for request {request_id}")
            if "error" in message:
                raise TamError(f"{method} failed: {message['error']}")
            return message["result"]

    def call(self, tool, arguments):
        result = self._request("tools/call", {"name": tool, "arguments": arguments}, CALL_TIMEOUT_S)
        text = "".join(part.get("text", "") for part in result.get("content", []))
        if result.get("isError"):
            raise TamError(f"{tool} failed: {text[:500]}")
        return json.loads(text)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        shutil.rmtree(self.memory_dir, ignore_errors=True)


class TamMemory:
    """Buffers the context stream, writes it as facts, answers top-k queries."""

    def __init__(self, command, project, show_recorded):
        self.tam = TamProcess(command)
        self.project = project
        self.show_recorded = show_recorded
        self.chunks = []
        self.flushed = False
        self.stats = {}

    def add(self, text):
        self.chunks.append(text)

    def flush(self):
        """Write every parsed fact, in context order. Idempotent."""
        from methods.agentmemory import parse_fact_lines

        if self.flushed:
            return self.stats
        facts = parse_fact_lines("".join(self.chunks))
        deduplicated = 0
        for fact in facts:
            saved = self.tam.call("memory_save", {"content": fact, "type": "fact", "project": self.project})
            if not saved.get("saved"):
                raise TamError(f"memory_save did not store a fact: {saved}")
            deduplicated += bool(saved.get("deduplicated"))
        self.stats = {"facts": len(facts), "deduplicated": deduplicated}
        self.flushed = True
        print(f"\ntotal-agent-memory flush: {self.stats}\n")
        return self.stats

    def query(self, text, k):
        found = self.tam.call("memory_recall", {
            "query": text, "project": self.project, "limit": k, "detail": "full",
        })
        # Every record is a `fact`, so this is one group, already in TAM's rank order.
        hits = [hit for group in (found.get("results") or {}).values() for hit in group]
        contents = []
        for hit in hits[:k]:
            content = hit.get("content") or ""
            if self.show_recorded and hit.get("created_at"):
                content = f"[recorded {hit['created_at']}] {content}"
            contents.append(content)
        return contents


def initialize_total_agent_memory_agent(agent, agent_config=None):
    config = agent_config or {}
    agent.retrieve_num = config["retrieve_num"]
    agent.context = ""
    agent.agent_start_time = time.time()
    command = os.environ.get("TAM_COMMAND", DEFAULT_COMMAND)
    show_recorded = bool(config.get("tam_show_recorded", True))
    agent.tam_memory = TamMemory(command, f"mab_{agent.sub_dataset}", show_recorded)
    print(f"\n\ntotal-agent-memory via `{command}`, show_recorded={show_recorded}\n\n")


def handle_total_agent_memory_agent(agent, message, memorizing, query_id, context_id):
    """Mirror `_handle_bm25_rag`: same query extraction, same reader assembly."""
    from methods.knowl import build_reader_messages, format_retrieval_memory_string
    from utils.templates import get_template

    memory = agent.tam_memory
    if memorizing:
        memory.add(message)
        return "Memorized"

    start_time = time.time()
    stats = memory.flush()
    memory_construction_time = time.time() - start_time

    retrieval_query = agent._extract_retrieval_query(message)
    contents = memory.query(retrieval_query, agent.retrieve_num)
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
    print(f"\ntotal-agent-memory stats: {stats}\n")

    return agent._create_standard_response(
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        memory_construction_time,
        query_time_len,
    )
