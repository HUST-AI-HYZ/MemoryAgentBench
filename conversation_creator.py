from utils.eval_other_utils import chunk_text_into_sentences
from utils.eval_data_utils import load_eval_data
from utils.templates import get_template

import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConversationCreator:
    """
    A class responsible for creating conversation data structures from various datasets.
    
    This class handles:
    - Loading dataset configurations
    - Processing contexts and questions/answers
    - Converting data into appropriate formats for agent consumption
    - Chunking text data for memory agents
    """

    def __init__(self, agent_config, dataset_config):
        """
        Initialize the ConversationCreator with agent and dataset configurations.
        
        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
        """
        # Store core configuration parameters
        self.dataset = dataset_config['dataset']
        self.max_test_samples = dataset_config['max_test_samples']
        self.context_max_length = dataset_config['context_max_length']
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        
        # Determine chunk size based on agent type
        self.chunk_size = self._determine_chunk_size(agent_config, dataset_config)
        
        # Process the dataset and create conversation structures
        self._load_and_process_dataset(dataset_config)

    def _determine_chunk_size(self, agent_config, dataset_config):
        """
        Determine the appropriate chunk size based on agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            dataset_config: Dataset configuration dictionary
            
        Returns:
            int: The chunk size to use for text processing
        """
        # Memory agents (mem0, letta, cognee) use agent-specific chunk size
        if agent_config.get('agent_chunk_size') is not None:
            assert any(agent_name in agent_config['agent_name'] 
                      for agent_name in ["mem0", "letta", "cognee", "zep", "amt"]), \
                   "agent_chunk_size should only be set for memory agents"
            
            chunk_size = agent_config['agent_chunk_size']
            print(f"\n\nUsing agent-specific chunk_size: {chunk_size}\n\n")
            return chunk_size
        else:
            # Default to dataset chunk size
            return dataset_config['chunk_size']

    def _load_and_process_dataset(self, dataset_config):
        """
        Load the dataset and process it into contexts and query-answer pairs.
        
        Args:
            dataset_config: Dataset configuration dictionary
        """
        logger.info(f"Running test on {self.sub_dataset}")

        # Load and convert dataset to processable format
        loaded_dataset = load_eval_data(dataset_config)
        dataset_items = self._convert_dataset_format(loaded_dataset)
        
        # Determine how many samples to process
        num_samples_to_process = min(len(dataset_items), self.max_test_samples)

        # Process each dataset item using list comprehension for better performance
        processed_items = [
            self._process_dataset_item(dataset_items[i]) 
            for i in range(num_samples_to_process)
        ]
        
        # Unpack contexts and query-answer pairs
        self.contexts, self.query_and_answers = zip(*processed_items) if processed_items else ([], [])
        self.contexts, self.query_and_answers = list(self.contexts), list(self.query_and_answers)

        if self.contexts:
            print(f"Dataset length: {len(self.contexts)}, each sample has {len(self.query_and_answers[0])} qa pairs")
        else:
            print("Dataset is empty - no samples found matching the filter criteria")
            raise ValueError(f"No samples found for sub_dataset: {self.sub_dataset}. Please check the dataset configuration.")

    def _convert_dataset_format(self, loaded_dataset):
        """
        Convert dataset from various formats to a consistent list format.
        
        Args:
            loaded_dataset: Raw dataset loaded from load_data()
            
        Returns:
            list: Dataset items in list format for consistent processing
        """
        # Handle both old format (direct list) and new HuggingFace format (dict with 'data' key)
        return (list(loaded_dataset['data']) 
                if isinstance(loaded_dataset, dict) and 'data' in loaded_dataset 
                else loaded_dataset)

    def _process_dataset_item(self, dataset_item):
        """
        Process a single dataset item to extract context and create query-answer pairs.
        
        Args:
            dataset_item: Single item from the dataset
            
        Returns:
            tuple: (context_text, list_of_qa_pairs)
        """
        # Extract and validate context
        context_text = dataset_item["context"]
        assert len(context_text) > 2000, f"Context too short: {len(context_text)} characters"
        
        # Extract all non-context fields for question generation
        question_data = {key: value for key, value in dataset_item.items() if key != "context"}
        
        # Create query-answer pairs from the question data
        qa_pairs = self._create_query_answer_pairs(question_data)
        
        return context_text, qa_pairs

    def _create_query_answer_pairs(self, question_data):
        """
        Create query-answer pairs from question data, handling both single and multiple Q&A.
        
        Args:
            question_data: Dictionary containing questions, answers, and metadata
            
        Returns:
            list: List of (query, answer, qa_pair_id) tuples
        """
        # Extract questions and answers, ensuring they are lists
        questions = self._ensure_list(question_data.get('questions', []))
        answers = self._ensure_list(question_data.get('answers', []))
        
        # Process question-answer pairs based on actual data structure
        if len(questions) > 1 and len(answers) > 1:
            # Multiple questions and answers - process each pair individually
            return [
                self._create_single_qa_pair(question_data, question, answer, i)
                for i, (question, answer) in enumerate(zip(questions, answers))
            ]
        else:
            # Single question or answer set - process as one unit
            return [self._create_single_qa_pair(
                question_data, questions[0] if questions else "", answers, 0
            )]

    def _create_single_qa_pair(self, question_data, question, answer, question_index):
        """
        Create a single query-answer pair with metadata.
        
        Args:
            question_data: Original question data dictionary
            question: The specific question text
            answer: The specific answer text
            question_index: Index of this question in the list
            
        Returns:
            tuple: (formatted_query, answer, qa_pair_id)
        """
        # Create question-specific metadata
        qa_metadata = self._create_qa_metadata(question_data, question, answer, question_index)
        
        # Generate the formatted query using template
        query_template = get_template(self.sub_dataset, 'query', self.agent_name)
        formatted_query = query_template.format(**qa_metadata)
        
        # Get qa_pair_id for this question if available
        qa_pair_id = qa_metadata.get('qa_pair_ids')
        
        return formatted_query, answer, qa_pair_id

    def _ensure_list(self, value):
        """
        Ensure a value is a list, converting single values to single-item lists.
        
        Args:
            value: Value that should be a list
            
        Returns:
            list: The value as a list
        """
        return value if isinstance(value, list) else [value]

    def _create_qa_metadata(self, question_data, question, answer, question_index):
        """
        Create metadata dictionary for a specific question-answer pair.
        
        Args:
            question_data: Original question data dictionary
            question: The specific question text
            answer: The specific answer text
            question_index: Index of this question in the list
            
        Returns:
            dict: Metadata dictionary for template formatting
        """
        # Start with a copy of the original question data
        qa_metadata = dict(question_data)
        
        # Set the specific question and answer
        qa_metadata.update({'question': question, 'answer': answer})
        
        # Process indexed fields
        indexed_fields = ['question_dates', 'question_types', 'question_ids', 'previous_events', 'qa_pair_ids']
        
        for field_name in indexed_fields:
            field_value = self._get_field_value(question_data, field_name, question_index)
            if field_value is not None:
                qa_metadata[field_name] = field_value
        
        # Handle source field specifically (can be nested under metadata)
        if 'source' not in qa_metadata:
            qa_metadata['source'] = question_data.get('metadata', {}).get('source', '')
        
        return qa_metadata

    def _get_field_value(self, question_data, field_name, question_index):
        """
        Get field value from either top level or nested metadata, handling indexing.
        
        Args:
            question_data: Original question data dictionary
            field_name: Name of the field to retrieve
            question_index: Index for list fields
            
        Returns:
            Field value or None if not found
        """
        # Check direct field first, then nested metadata
        field_value = (question_data.get(field_name) or 
                      question_data.get('metadata', {}).get(field_name))
        
        if field_value is None:
            return None
        
        # Use indexed value if it's a list with enough entries, otherwise use the whole value
        return (field_value[question_index] 
                if isinstance(field_value, list) and question_index < len(field_value)
                else field_value)

    def get_chunks(self):
        """
        Get text chunks for all contexts, suitable for memory agent processing.
        
        Returns:
            list: List of lists, where each inner list contains text chunks for one context
        """
        all_context_chunks = [
            chunk_text_into_sentences(context, chunk_size=self.chunk_size)
            for context in self.contexts
        ]
        
        # Validate the output structure
        self._validate_chunks_structure(all_context_chunks)
        return all_context_chunks

    def _validate_chunks_structure(self, chunks):
        """
        Validate that the chunks have the expected structure.
        
        Args:
            chunks: The chunks structure to validate
            
        Raises:
            AssertionError: If the structure is not as expected
        """
        assert isinstance(chunks, list), "Chunks should be a list"
        assert len(chunks) > 0, "Chunks should not be empty"
        assert isinstance(chunks[0], list), "Each context should have a list of chunks"
        assert isinstance(chunks[0][0], str), "Each chunk should be a string"

    def get_query_and_answers(self):
        """
        Get the processed query-answer pairs for all contexts.
        
        Returns:
            list: List of lists, where each inner list contains (query, answer, qa_pair_id) tuples for one context
        """
        # Validate the output structure
        self._validate_qa_structure(self.query_and_answers)
        return self.query_and_answers

    def _validate_qa_structure(self, query_and_answers):
        """
        Validate that the query-answer structure is correct.
        
        Args:
            query_and_answers: The query-answer structure to validate
            
        Raises:
            AssertionError: If the structure is not as expected
        """
        assert isinstance(query_and_answers, list), "Query-answers should be a list"
        assert len(query_and_answers) > 0, "Query-answers should not be empty"
        assert isinstance(query_and_answers[0], list), "Each context should have a list of QA pairs"
        # Each QA pair should be a tuple of (query, answer, qa_pair_id)
        if len(query_and_answers[0]) > 0:
            assert len(query_and_answers[0][0]) == 3, "Each QA pair should be a tuple of (query, answer, qa_pair_id)"