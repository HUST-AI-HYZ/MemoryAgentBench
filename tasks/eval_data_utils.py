from datasets import load_dataset
from tasks.eval_other_utils import calculate_metrics, parse_output
import json
import os
import random

import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# HUGGING FACE DATASET LOADING
# ============================================================================

def load_data_huggingface(dataset_name, sub_dataset_source, max_test_samples=None, seed=42):
    """    
    Args:
        dataset_name: The dataset name (Accurate_Retrieval, Test_Time_Learning, 
                     Long_Range_Understanding, Conflict_Resolution)
        sub_dataset_source: The sub_dataset name used to filter by source field
        max_test_samples: Maximum number of test samples to load
        seed: Random seed for sampling
    
    Returns:
        Dictionary with processed data
    """
    print(f"Loading {sub_dataset_source} from Hugging Face dataset: ai-hyz/MemoryAgentBench")
    
    # Configuration for Hugging Face dataset
    huggingface_dataset_name = "ai-hyz/MemoryAgentBench"
    
    # Supported dataset splits (identity mapping)
    supported_splits = {
        "Accurate_Retrieval", "Test_Time_Learning", 
        "Long_Range_Understanding", "Conflict_Resolution"
    }
    
    # Validate dataset name
    if dataset_name not in supported_splits:
        raise ValueError(f"Unknown dataset {dataset_name}. Available splits: {sorted(supported_splits)}")
    
    split_name = dataset_name
    
    # Load, filter, and process dataset
    dataset = _load_and_filter_dataset(huggingface_dataset_name, split_name, sub_dataset_source, max_test_samples, seed)
    processed_dataset = _process_qa_list_fields(dataset)
    
    return {"data": processed_dataset}



def _load_and_filter_dataset(dataset_name, split_name, source_filter, max_samples, seed):
    """Load dataset from HuggingFace and apply filtering and sampling."""
    try:
        # Load the specific split from HuggingFace
        raw_data = load_dataset(dataset_name, split=split_name)
        print(f"Loaded {len(raw_data)} samples from {split_name}")
        
        # Filter by source to match the sub_dataset exactly (source is now in metadata)
        original_length = len(raw_data)
        filtered_data = raw_data.filter(lambda sample: sample.get("metadata", {}).get("source", "") == source_filter)
        print(f"Filtered to {len(filtered_data)} samples matching source '{source_filter}' "
              f"(from {original_length} total)")
        
        # Apply max_test_samples limit if specified
        if max_samples is not None and len(filtered_data) > max_samples:
            filtered_data = filtered_data.select(range(max_samples))
            print(f"Subsampled to {max_samples} samples")
        
        return filtered_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise ValueError(f"Split {split_name} not found in dataset.")


def _process_qa_list_fields(dataset):
    """
    Process the dataset to ensure Q&A pairs and related fields are properly formatted as lists.
    
    Args:
        dataset: HuggingFace dataset object
        
    Returns:
        HuggingFace dataset with processed list fields
    """
    # Convert back to HuggingFace dataset format
    from datasets import Dataset as HFDataset
    return HFDataset.from_list([_process_single_sample_qa_lists(sample) for sample in dataset])


def _process_single_sample_qa_lists(sample):
    """
    Process a single sample to ensure all Q&A related fields are properly formatted as lists.
    
    Args:
        sample: Single data sample dictionary
        
    Returns:
        Processed sample with list-formatted fields
    """
    # Process main Q&A fields
    metadata = sample.get("metadata", {})
    
    # Define metadata fields to process
    metadata_fields = ["question_dates", "question_types", "question_ids", "previous_events", "uuids", "demo"]
    
    # Create processed sample with standardized list fields
    processed_sample = dict(sample)
    processed_sample.update({
        "questions": _ensure_field_is_list(sample["questions"]),
        "answers": _ensure_field_is_list(sample["answers"]),
        "source": metadata.get("source", ""),
        **{field: _ensure_field_is_list(metadata.get(field, [])) for field in metadata_fields}
    })
    
    return processed_sample


def _ensure_field_is_list(field_value):
    """
    Ensure a field value is properly formatted as a list.
    
    Args:
        field_value: Value that should be converted to list format
        
    Returns:
        List representation of the field value
    """
    if isinstance(field_value, list):
        return field_value
    elif field_value:
        # Single value (string or other) - wrap in list
        return [field_value]
    else:
        # Empty or None value
        return []


# ============================================================================
# LOCAL TEXT DATA LOADING
# ============================================================================

def load_data_localtxt(dataset_name, sub_dataset_source, max_test_samples=None, seed=42):
    """
    Load data from local text/JSON files.
    
    Args:
        dataset_name: The main dataset name (used to determine file path)
        sub_dataset_source: The sub_dataset name (used to find specific files)
        max_test_samples: Maximum number of test samples to load
        seed: Random seed for sampling
    
    Returns:
        Dictionary with processed data
    """
    print(f"Loading {sub_dataset_source} from local files")
    
    # Load Q&A and context data
    qa_data = _load_local_json_file(_construct_local_qa_file_path(sub_dataset_source))
    context_data = _load_local_context_file(_construct_local_context_file_path(sub_dataset_source))
    
    # Apply sampling and convert to expected dataset format
    sampled_qa_data = _apply_sampling_to_local_data(qa_data, max_test_samples, seed)
    processed_dataset = _convert_local_data_to_dataset_format(sampled_qa_data, context_data, sub_dataset_source)
    
    return {"data": processed_dataset}


def _construct_local_file_path(sub_dataset_source, file_type="qa"):
    """Construct file path for local data files."""
    base_dir = "raw_dataset/new_processed_data"
    
    if file_type == "context":
        return os.path.join(base_dir, sub_dataset_source, f"{sub_dataset_source}_context.txt")
    
    # For Q&A files, try multiple patterns
    primary_path = os.path.join(base_dir, sub_dataset_source, f"{sub_dataset_source}_pairs.json")
    
    if os.path.exists(primary_path):
        return primary_path
    
    # Alternative patterns
    alt_patterns = [
        f"{base_dir}/{sub_dataset_source}/pairs.json",
        f"{base_dir}/{sub_dataset_source}.json",
        f"{base_dir}/{sub_dataset_source}/{sub_dataset_source}.json"
    ]
    
    for alt_path in alt_patterns:
        if os.path.exists(alt_path):
            return alt_path
    
    raise FileNotFoundError(f"Could not find local Q&A data file for {sub_dataset_source}. "
                          f"Checked: {primary_path} and alternatives")


def _construct_local_qa_file_path(sub_dataset_source):
    """Construct the file path for local Q&A data files."""
    return _construct_local_file_path(sub_dataset_source, "qa")


def _construct_local_context_file_path(sub_dataset_source):
    """Construct the file path for local context data files."""
    return _construct_local_file_path(sub_dataset_source, "context")


def _load_local_file(file_path, file_type="json"):
    """Load and parse local data files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_type == "json":
                data = json.load(f)
                print(f"Loaded {len(data)} samples from {file_path}")
                return data
            else:  # text file
                content = f.read().strip()
                print(f"Loaded context from {file_path} ({len(content)} characters)")
                return content
    except Exception as e:
        raise ValueError(f"Error loading local file {file_path}: {e}")


def _load_local_json_file(file_path):
    """Load and parse JSON file."""
    return _load_local_file(file_path, "json")


def _load_local_context_file(file_path):
    """Load and parse context data from text file."""
    return _load_local_file(file_path, "text")


def _apply_sampling_to_local_data(data, max_samples, seed):
    """Apply sampling to local data if specified."""
    if max_samples is None or len(data) <= max_samples:
        return data
    
    random.seed(seed)
    sampled_data = random.sample(data, max_samples)
    print(f"Subsampled to {max_samples} samples")
    return sampled_data


def _convert_local_data_to_dataset_format(qa_data, context_data, source_name):
    """
    Convert local data to the expected dataset format.
    
    Args:
        qa_data: List of dictionaries from Q&A JSON file
        context_data: String containing the context text
        source_name: Name of the data source
        
    Returns:
        HuggingFace dataset with standardized fields
    """
    def format_qa_item(qa_item):
        """Format a single Q&A item."""
        answers = qa_item.get("answer", [])
        
        # Ensure answers are in list format
        if isinstance(answers, str):
            answers = [answers]
        elif not isinstance(answers, list):
            answers = [str(answers)]
        
        return {
            "questions": [qa_item.get("question", "")],
            "answers": answers,
            "source": source_name,
            "question_dates": [],
            "question_types": [],
            "question_ids": [str(qa_item.get("index", ""))],
            "previous_events": [],
            "context_length": qa_item.get("context_length", len(context_data)),
            "original_index": qa_item.get("original_index", qa_item.get("index", 0)),
            "context": context_data  # Use the entire context for all Q&A pairs
        }
    
    # Convert to HuggingFace dataset format
    from datasets import Dataset as HFDataset
    return HFDataset.from_list([format_qa_item(qa_item) for qa_item in qa_data])


# ============================================================================
# MAIN DATA LOADING INTERFACE
# ============================================================================

def load_eval_data(dataset_config):
    """
    Main interface for loading dataset based on configuration.
    
    Args:
        dataset_config: Dictionary containing dataset configuration parameters
        
    Returns:
        Loaded and processed dataset
    """
    # Extract configuration parameters
    config_params = (
        dataset_config['dataset'], dataset_config['sub_dataset'],
        dataset_config["max_test_samples"], dataset_config["seed"]
    )
    main_dataset_name, sub_dataset_name, max_test_samples, random_seed = config_params
    
    print(f"Dataset: {sub_dataset_name}")
    
    # Load data based on dataset type
    supported_hf_datasets = {
        'Accurate_Retrieval', 'Test_Time_Learning', 
        'Long_Range_Understanding', 'Conflict_Resolution'
    }
    
    # Check if it's a HuggingFace dataset
    if main_dataset_name in supported_hf_datasets:
        return load_data_huggingface(main_dataset_name, sub_dataset_name, max_test_samples, random_seed)
    
    # Otherwise, try to load from local files
    try:
        return load_data_localtxt(main_dataset_name, sub_dataset_name, max_test_samples, random_seed)
    except FileNotFoundError as e:
        # If local file not found, provide helpful error message
        raise ValueError(f"Dataset '{sub_dataset_name}' not found. "
                       f"Supported HuggingFace datasets: {supported_hf_datasets}. "
                       f"For local datasets, ensure files exist in raw_dataset/new_processed_data/. "
                       f"Error: {e}")


# ============================================================================
# CHAT FORMATTING UTILITIES
# ============================================================================

def format_chat(message, include_system=True, system_message="You are a helpful assistant."):
    """
    Format a message into chat format for language model consumption.
    
    Args:
        message: The user message content
        include_system: Whether to include system message in the chat
        system_message: The system message content
        
    Returns:
        List of message dictionaries in chat format
    """
    chat_messages = [{"role": "user", "content": message}]
    
    if include_system:
        chat_messages.insert(0, {"role": "system", "content": system_message})
    
    return chat_messages










