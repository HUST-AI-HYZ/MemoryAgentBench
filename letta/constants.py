import os
from logging import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARN, WARNING

LETTA_DIR = os.path.join(os.path.expanduser("~"), ".letta")
LETTA_TOOL_EXECUTION_DIR = os.path.join(LETTA_DIR, "tool_execution_dir")

ADMIN_PREFIX = "/v1/admin"
API_PREFIX = "/v1"
OPENAI_API_PREFIX = "/openai"

COMPOSIO_ENTITY_ENV_VAR_KEY = "COMPOSIO_ENTITY"
COMPOSIO_TOOL_TAG_NAME = "composio"

MCP_CONFIG_NAME = "mcp_config.json"
MCP_TOOL_TAG_NAME_PREFIX = "mcp"  # full format, mcp:server_name

LETTA_CORE_TOOL_MODULE_NAME = "letta.functions.function_sets.base"
LETTA_MULTI_AGENT_TOOL_MODULE_NAME = "letta.functions.function_sets.multi_agent"

# String in the error message for when the context window is too large
# Example full message:
# This model's maximum context length is 8192 tokens. However, your messages resulted in 8198 tokens (7450 in the messages, 748 in the functions). Please reduce the length of the messages or functions.
OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING = "maximum context length"

# System prompt templating
IN_CONTEXT_MEMORY_KEYWORD = "CORE_MEMORY"

# OpenAI error message: Invalid 'messages[1].tool_calls[0].id': string too long. Expected a string with maximum length 29, but got a string with length 36 instead.
TOOL_CALL_ID_MAX_LEN = 29

# minimum context window size
MIN_CONTEXT_WINDOW = 4096

# embeddings
MAX_EMBEDDING_DIM = 4096  # maximum supported embeding size - do NOT change or else DBs will need to be reset
DEFAULT_EMBEDDING_CHUNK_SIZE = 300

# tokenizers
EMBEDDING_TO_TOKENIZER_MAP = {
    "text-embedding-ada-002": "cl100k_base",
}
EMBEDDING_TO_TOKENIZER_DEFAULT = "cl100k_base"


DEFAULT_LETTA_MODEL = "gpt-4"  # TODO: fixme
DEFAULT_PERSONA = "sam_pov"
DEFAULT_HUMAN = "basic"
DEFAULT_PRESET = "memgpt_chat"

# Base tools that cannot be edited, as they access agent state directly
# Note that we don't include "conversation_search_date" for now
BASE_TOOLS = ["send_message", "conversation_search", "archival_memory_insert", "archival_memory_search"]
# Base memory tools CAN be edited, and are added by default by the server
BASE_MEMORY_TOOLS = ["core_memory_append", "core_memory_replace"]
# Base tools if the memgpt agent has enable_sleeptime on
BASE_SLEEPTIME_CHAT_TOOLS = ["send_message", "conversation_search", "archival_memory_search"]
# Base memory tools for sleeptime agent
BASE_SLEEPTIME_TOOLS = [
    "memory_replace",
    "memory_insert",
    "memory_rethink",
    "memory_finish_edits",
    # "archival_memory_insert",
    # "archival_memory_search",
    # "conversation_search",
]
# Multi agent tools
MULTI_AGENT_TOOLS = ["send_message_to_agent_and_wait_for_reply", "send_message_to_agents_matching_tags", "send_message_to_agent_async"]
# Set of all built-in Letta tools
LETTA_TOOL_SET = set(BASE_TOOLS + BASE_MEMORY_TOOLS + MULTI_AGENT_TOOLS + BASE_SLEEPTIME_TOOLS)

# The name of the tool used to send message to the user
# May not be relevant in cases where the agent has multiple ways to message to user (send_imessage, send_discord_mesasge, ...)
# or in cases where the agent has no concept of messaging a user (e.g. a workflow agent)
DEFAULT_MESSAGE_TOOL = "send_message"
DEFAULT_MESSAGE_TOOL_KWARG = "message"

PRE_EXECUTION_MESSAGE_ARG = "pre_exec_msg"

REQUEST_HEARTBEAT_PARAM = "request_heartbeat"


# Structured output models
STRUCTURED_OUTPUT_MODELS = {"gpt-4o", "gpt-4o-mini"}

# LOGGER_LOG_LEVEL is use to convert Text to Logging level value for logging mostly for Cli input to setting level
LOGGER_LOG_LEVELS = {"CRITICAL": CRITICAL, "ERROR": ERROR, "WARN": WARN, "WARNING": WARNING, "INFO": INFO, "DEBUG": DEBUG, "NOTSET": NOTSET}

FIRST_MESSAGE_ATTEMPTS = 10

INITIAL_BOOT_MESSAGE = "Boot sequence complete. Persona activated."
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT = "Bootup sequence complete. Persona activated. Testing messaging functionality."
STARTUP_QUOTES = [
    "I think, therefore I am.",
    "All those moments will be lost in time, like tears in rain.",
    "More human than human is our motto.",
]
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG = STARTUP_QUOTES[2]

CLI_WARNING_PREFIX = "Warning: "

ERROR_MESSAGE_PREFIX = "Error"

NON_USER_MSG_PREFIX = "[This is an automated system message hidden from the user] "

CORE_MEMORY_LINE_NUMBER_WARNING = (
    "# NOTE: Line numbers shown below are to help during editing. Do NOT include line number prefixes in your memory edit tool calls."
)


# Constants to do with summarization / conversation length window
# The max amount of tokens supported by the underlying model (eg 8k for gpt-4 and Mistral 7B)
LLM_MAX_TOKENS = {
    "DEFAULT": 8192,
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
    ## OpenAI models: https://platform.openai.com/docs/models/overview
    "gpt-4.1": 1047576,
    "gpt-4.1-2025-04-14": 1047576,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    # gpt-4.5-preview
    "gpt-4.5-preview": 128000,
    "gpt-4.5-preview-2025-02-27": 128000,
    # "o1-preview
    "chatgpt-4o-latest": 128000,
    # "o1-preview-2024-09-12
    "gpt-4o-2024-08-06": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo-instruct": 16385,
    "gpt-4-0125-preview": 128000,
    "gpt-3.5-turbo-0125": 16385,
    # "babbage-002": 128000,
    # "davinci-002": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    # "gpt-4o-realtime-preview-2024-10-01
    "gpt-4-turbo": 8192,
    "gpt-4o-2024-05-13": 128000,
    # "o1-mini
    # "o1-mini-2024-09-12
    # "gpt-3.5-turbo-instruct-0914
    "gpt-4o-mini": 128000,
    # "gpt-4o-realtime-preview
    "gpt-4o-mini-2024-07-18": 128000,
    # gpt-4
    "gpt-4-1106-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    "gpt-4-0314": 8192,  # legacy
    "gpt-4-32k-0314": 32768,  # legacy
    # gpt-3.5
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-0613": 4096,  # legacy
    "gpt-3.5-turbo-16k-0613": 16385,  # legacy
    "gpt-3.5-turbo-0301": 4096,  # legacy
}
# The error message that Letta will receive
# MESSAGE_SUMMARY_WARNING_STR = f"Warning: the conversation history will soon reach its maximum length and be trimmed. Make sure to save any important information from the conversation to your memory before it is removed."
# Much longer and more specific variant of the prompt
MESSAGE_SUMMARY_WARNING_STR = " ".join(
    [
        f"{NON_USER_MSG_PREFIX}The conversation history will soon reach its maximum length and be trimmed.",
        "Do NOT tell the user about this system alert, they should not know that the history is reaching max length.",
        "If there is any important new information or general memories about you or the user that you would like to save, you should save that information immediately by calling function core_memory_append, core_memory_replace, or archival_memory_insert.",
        # "Remember to pass request_heartbeat = true if you would like to send a message immediately after.",
    ]
)
DATA_SOURCE_ATTACH_ALERT = (
    "[ALERT] New data was just uploaded to archival memory. You can view this data by calling the archival_memory_search tool."
)

# The ackknowledgement message used in the summarize sequence
MESSAGE_SUMMARY_REQUEST_ACK = "Understood, I will respond with a summary of the message (and only the summary, nothing else) once I receive the conversation history. I'm ready."

# Maximum length of an error message
MAX_ERROR_MESSAGE_CHAR_LIMIT = 500

# Default memory limits
CORE_MEMORY_PERSONA_CHAR_LIMIT: int = 5000
CORE_MEMORY_HUMAN_CHAR_LIMIT: int = 5000
CORE_MEMORY_BLOCK_CHAR_LIMIT: int = 5000

# Function return limits
FUNCTION_RETURN_CHAR_LIMIT = 6000  # ~300 words
BASE_FUNCTION_RETURN_CHAR_LIMIT = 1000000  # very high (we rely on implementation)

MAX_PAUSE_HEARTBEATS = 360  # in min

MESSAGE_CHATGPT_FUNCTION_MODEL = "gpt-3.5-turbo"
MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE = "You are a helpful assistant. Keep your responses short and concise."

#### Functions related

# REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}request_heartbeat == true"
REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function called using request_heartbeat=true, returning control"
# FUNC_FAILED_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function call failed"
FUNC_FAILED_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function call failed, returning control"


RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 10

MAX_FILENAME_LENGTH = 255
RESERVED_FILENAMES = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"}
