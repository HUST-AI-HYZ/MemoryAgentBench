import os
import json
import time
import yaml
import shutil
from collections import defaultdict
from conversation_creator import ConversationCreator
from agent import AgentWrapper
from tqdm import tqdm
from utils.eval_other_utils import metrics_summarization


# ============================================================================
# MAIN WORKFLOW FUNCTIONS (in typical execution order)
# ============================================================================

def setup_configs_and_directories(command_line_args):
    """
    Setup configurations, handle ablations, cleanup, and create output directories.
    
    Args:
        command_line_args: Parsed command line arguments
        
    Returns:
        tuple: (agent_config, dataset_config, output_file_path)
    """
    # Load configuration files
    agent_config = _load_yaml_config(command_line_args.agent_config)
    dataset_config = _load_yaml_config(command_line_args.dataset_config)
    
    # Apply ablation study parameters if specified
    _apply_ablation_parameters(command_line_args, agent_config, dataset_config)
    
    # Clean up previous agent data if necessary
    _cleanup_agent_directories(agent_config)
    
    # Create output directory and file path
    output_file_path = _create_output_path(agent_config, dataset_config)
    
    return agent_config, dataset_config, output_file_path


def create_agent_and_fetch_data(agent_config, dataset_config):
    """
    Create conversation creator and fetch chunks and query_and_answers.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        
    Returns:
        tuple: (start_time, all_context_chunks, all_query_answer_pairs)
    """
    start_time = time.time()
    
    # Create conversation creator to handle data loading and processing
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    
    # Fetch processed chunks and query-answer pairs
    return start_time, conversation_creator.get_chunks(), conversation_creator.get_query_and_answers()


def load_existing_results(output_file_path, dataset_config, all_query_answer_pairs):
    """
    Load existing results from output file and initialize variables.
    
    Args:
        output_file_path: Path to the output results file
        dataset_config: Configuration dictionary for the dataset
        all_query_answer_pairs: List of query-answer pairs for all contexts
        
    Returns:
        tuple: (metrics, results, last_completed_context_id, last_completed_query_id)
    """
    if not os.path.exists(output_file_path):
        return defaultdict(list), [], 0, 0
    
    # Load existing results from file
    with open(output_file_path, "r") as file:
        saved_output = json.load(file)
        
    # Initialize data structures
    metrics, results = defaultdict(list), []
    
    # Process each saved result entry
    for saved_data_entry in saved_output['data']:
        query = saved_data_entry['query']
        
        # Handle both list and string answer formats
        answer = (saved_data_entry['answer'][0] 
                 if isinstance(saved_data_entry['answer'], list) 
                 else saved_data_entry['answer'])
        
        # Reconstruct output format expected by metrics_summarization
        reconstructed_output = {
            "output": saved_data_entry['output'],
            "input_len": saved_data_entry['input_len'],
            "output_len": saved_data_entry['output_len'],
            "memory_construction_time": saved_data_entry.get('memory_construction_time', 0),
            "query_time_len": saved_data_entry['query_time_len'],
        }
        
        # Extract existing identifiers
        existing_query_id = saved_data_entry.get('query_id')
        existing_qa_pair_id = saved_data_entry.get('qa_pair_id')
        
        metrics, results = metrics_summarization(
            reconstructed_output, query, answer, dataset_config, 
            metrics, results, existing_query_id, existing_qa_pair_id
        )
    
    # Calculate the last completed context ID
    total_queries_processed = len(results)
    last_completed_context_id = _calculate_last_completed_context_id(
        all_query_answer_pairs, total_queries_processed
    )
    
    return metrics, results, last_completed_context_id, total_queries_processed


def generate_agent_save_folder(agent_config, dataset_config, current_context_index):
    """
    Generate the agent save folder path based on agent type and configuration.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        current_context_index: Index of the current context being processed
        
    Returns:
        str: Path to the agent save folder
    """
    agent_name = agent_config['agent_name']
    
    # Generate base path based on agent type
    if any(agent_type in agent_name for agent_type in ["mem0", "cognee", "letta", "zep", "amt"]):
        base_path = _generate_memory_agent_base_path(agent_config, dataset_config)
        return f"{base_path}/exp_{current_context_index}"
    elif "rag" in agent_name:
        return _generate_rag_agent_path(agent_config, dataset_config, current_context_index)
    else:
        return _generate_default_agent_path(agent_config, dataset_config, current_context_index)


def initialize_and_memorize_agent(agent_config, dataset_config, agent_save_folder, 
                                 context_chunks, current_context_index, total_contexts_count):
    """
    Initialize agent and handle memorization if needed.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        agent_save_folder: Path to folder where agent state is saved
        context_chunks: List of text chunks for the current context
        current_context_index: Index of the current context
        total_contexts_count: Total number of contexts to process
        
    Returns:
        AgentWrapper: Initialized agent ready for querying
    """
    # Initialize the agent wrapper
    agent = AgentWrapper(agent_config, dataset_config, load_agent_from=agent_save_folder)
    
    # Handle memorization or loading based on whether saved state exists
    if os.path.exists(agent_save_folder):
        agent.load_agent()
        print("\n\n Agent loaded...\n\n")
    else:
        _memorize_context_chunks(agent, context_chunks, current_context_index, total_contexts_count)
        agent.save_agent()
        
    return agent


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def _load_yaml_config(config_file_path):
    """Load and return YAML configuration from file."""
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)


def _apply_ablation_parameters(command_line_args, agent_config, dataset_config):
    """Apply ablation study parameters to override default configurations."""
    # Handle chunk size ablation
    if command_line_args.chunk_size_ablation > 0:
        _apply_chunk_size_ablation(command_line_args, agent_config, dataset_config)
    
    # Handle max test queries ablation
    if command_line_args.max_test_queries_ablation > 0:
        dataset_config['max_test_queries'] = command_line_args.max_test_queries_ablation
        print(f"\n\nUsing max_test_queries: {dataset_config['max_test_queries']}\n\n")


def _apply_chunk_size_ablation(command_line_args, agent_config, dataset_config):
    """Apply chunk size ablation based on agent type."""
    new_chunk_size = command_line_args.chunk_size_ablation
    
    # Check if this is a memory agent that uses agent_chunk_size
    if any(agent_name in agent_config['agent_name'] for agent_name in ['mem0', 'letta', 'cognee', 'zep', 'amt']):
        agent_config['agent_chunk_size'] = new_chunk_size
        dataset_config['chunk_size'] = new_chunk_size
        print(f"\n\nUsing agent chunk_size: {agent_config['agent_chunk_size']}\n\n")
    else:
        dataset_config['chunk_size'] = new_chunk_size
        print(f"\n\nUsing new chunk_size: {dataset_config['chunk_size']}\n\n")


def _cleanup_agent_directories(agent_config):
    """Clean up previous agent data directories if necessary."""
    if agent_config['agent_name'] == 'cognee':
        for directory_path in ['./cognee/.data_storage/data', './cognee/.cognee_system/databases']:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)


# ============================================================================
# OUTPUT PATH GENERATION HELPERS
# ============================================================================

def _create_output_path(agent_config, dataset_config):
    """
    Create output directory and return the output file path.
    
    Args:
        agent_config: Configuration dictionary for the agent
        dataset_config: Configuration dictionary for the dataset
        
    Returns:
        str: Path to the output results file
    """
    # Generate name tag based on agent type and configuration
    name_tag = _generate_output_name_tag(agent_config, dataset_config)
    
    # Create output directory for this dataset
    output_directory = os.path.join(agent_config['output_dir'], dataset_config['dataset'])
    os.makedirs(output_directory, exist_ok=True)
    
    # Create complete output file path
    return os.path.join(output_directory, f"{name_tag}_results.json")


def _generate_output_name_tag(agent_config, dataset_config):
    """Generate a descriptive name tag for output files based on configuration."""
    def safe_get(config_dict, key, default="unknown"):
        """Helper function to safely get config values and convert to string."""
        value = config_dict.get(key, default)
        return str(value) if value is not None else default
    
    # Base components for all agents
    base_components = [
        safe_get(dataset_config, 'sub_dataset'),
        safe_get(dataset_config, 'tag'),
        f"in{safe_get(dataset_config, 'context_max_length')}",
        f"size{safe_get(dataset_config, 'generation_max_length')}",
        f"shots{safe_get(dataset_config, 'shots')}",
        f"max_samples{safe_get(dataset_config, 'max_test_samples')}"
    ]
    
    # Agent-specific components
    agent_name = safe_get(agent_config, 'agent_name')
    agent_components = []
    
    if "letta" in agent_name:
        agent_components = [
            f"chunk{safe_get(agent_config, 'agent_chunk_size')}",
            f"mode{safe_get(agent_config, 'letta_mode')}"
        ]
    elif any(agent_type in agent_name for agent_type in ["mem0", "cognee", "zep", "amt"]):
        agent_components = [
            f"k{safe_get(agent_config, 'retrieve_num')}",
            f"chunk{safe_get(agent_config, 'agent_chunk_size')}"
        ]
    elif "rag" in agent_name:
        agent_components = [
            f"k{safe_get(agent_config, 'retrieve_num')}",
            f"chunk{safe_get(dataset_config, 'chunk_size')}"
        ]
    
    return _sanitize_path_segment("_".join(base_components + agent_components))


# ============================================================================
# RESULTS LOADING HELPERS
# ============================================================================

def _calculate_last_completed_context_id(all_query_answer_pairs, total_queries_processed):
    """
    Calculate how many complete contexts have been processed based on total queries.
    
    Args:
        all_query_answer_pairs: List of query-answer pairs for all contexts
        total_queries_processed: Total number of queries that have been processed
        
    Returns:
        int: Number of completely processed contexts
    """
    queries_counted = 0
    
    for context_id, query_answer_pairs in enumerate(all_query_answer_pairs):
        if queries_counted + len(query_answer_pairs) <= total_queries_processed:
            queries_counted += len(query_answer_pairs)
        else:
            return context_id
            
    return len(all_query_answer_pairs)


# ============================================================================
# AGENT FOLDER GENERATION HELPERS
# ============================================================================

def _sanitize_path_segment(value):
    """Replace characters that are invalid in Windows path segments."""
    if value is None:
        return ""
    text = str(value)
    for ch in '<>:"/\\|?*':
        text = text.replace(ch, '_')
    return text


def _generate_memory_agent_base_path(agent_config, dataset_config):
    """Generate base path for memory agents (letta, mem0, cognee, zep, amt)."""
    agent_name = agent_config['agent_name']
    sub_dataset = _sanitize_path_segment(dataset_config['sub_dataset'])
    model = _sanitize_path_segment(agent_config['model'])
    base_path = f"./agents/{agent_name}_{sub_dataset}_chunk{agent_config['agent_chunk_size']}_model{model}"

    return (f"{base_path}_mode{_sanitize_path_segment(agent_config['letta_mode'])}"
            if "letta" in agent_name else base_path)


def _generate_rag_agent_path(agent_config, dataset_config, current_context_index):
    """Generate path for RAG agents."""
    sub_dataset = _sanitize_path_segment(dataset_config['sub_dataset'])
    model = _sanitize_path_segment(agent_config['model'])
    return (f"./agents/{agent_config['agent_name']}_{sub_dataset}"
            f"_k{agent_config['retrieve_num']}_chunk{dataset_config['chunk_size']}"
            f"_model{model}/exp_{current_context_index}")


def _generate_default_agent_path(agent_config, dataset_config, current_context_index):
    """Generate path for default agents."""
    sub_dataset = _sanitize_path_segment(dataset_config['sub_dataset'])
    return (f"./agents/{agent_config['agent_name']}_{sub_dataset}"
            f"/exp_{current_context_index}")


# ============================================================================
# AGENT INITIALIZATION HELPERS
# ============================================================================

def _memorize_context_chunks(agent, context_chunks, current_context_index, total_contexts_count):
    """Handle the memorization process for context chunks."""
    print("\n\n Agent Memorizing...\n\n")
    
    progress_description = f"Processing experiments {current_context_index + 1}/{total_contexts_count}"
    
    for chunk in tqdm(context_chunks, total=len(context_chunks), desc=progress_description):
        agent.send_message(chunk, memorizing=True)

    
