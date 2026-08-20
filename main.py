import os
import yaml
import dotenv
import time
import json
from agent import AgentWrapper
from argparse import ArgumentParser
from conversation_creator import ConversationCreator
from initialization import (
    load_existing_results, 
    create_agent_and_fetch_data, 
    setup_configs_and_directories, 
    generate_agent_save_folder, 
    initialize_and_memorize_agent
)
from tqdm import tqdm
from collections import defaultdict
import logging
import numpy as np
from utils.eval_other_utils import metrics_summarization

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
dotenv.load_dotenv()


def parse_command_line_arguments():
    """Parse and return command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--agent_config', type=str, default='configs/model_conf/client_long_context.yaml',
                       help='Path to agent configuration file')
    parser.add_argument('--dataset_config', type=str, default='configs/data_conf/HELMET_InfBench.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--chunk_size_ablation', type=int, default=0,
                       help='Override chunk size for ablation studies (0 = use config default)')
    parser.add_argument('--max_test_queries_ablation', type=int, default=0,
                       help='Limit maximum test queries for ablation studies (0 = no limit)')
    parser.add_argument('--force', action='store_true', default=False,
                       help='Force re-run even if results already exist')
    return parser.parse_args()


def should_skip_context(force_rerun, context_index, last_processed_context_id):
    """Determine if we should skip a context that has already been processed."""
    return not force_rerun and context_index < last_processed_context_id


def should_skip_query(query_index, last_processed_query_id):
    """Determine if we should skip a query that has already been processed."""
    return query_index < last_processed_query_id


def has_reached_query_limit(max_queries, current_query_index):
    """Determine if we should stop processing due to reaching the query limit."""
    return max_queries > 0 and current_query_index >= max_queries


def save_results_to_file(output_path, agent_config, dataset_config, results, metrics, time_cost_list, start_time):
    """Save current results to the output file."""
    # Calculate averaged metrics for logging
    averaged_metrics = {
        key: np.mean(values) * (1 if ("_len" in key) or ("_time" in key) else 100) 
        for key, values in metrics.items()
    }
    
    # Log current metrics
    for key, value in averaged_metrics.items():
        logger.info(f"{key}: {value:.02f}")
    
    # Prepare output data structure
    time_cost_list.append(time.time() - start_time)
    output_data = {
        "agent_config": agent_config,
        "dataset_config": dataset_config,
        "data": results,
        "metrics": metrics,
        "time_cost": time_cost_list,
        "averaged_metrics": averaged_metrics,
    }
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)
    logger.info(f"Results saved at {output_path}")


def process_single_query(agent, query, answer, dataset_config, metrics, results, 
                        query_index, context_index, qa_pair_id=None):
    """Process a single query and update metrics and results."""
    # Send query to agent and get response
    agent_output = agent.send_message(query, memorizing=False, query_id=query_index, context_id=context_index)
    
    # Calculate metrics and update results
    return metrics_summarization(agent_output, query, answer, dataset_config, metrics, results, query_index, qa_pair_id)


def unpack_query_data(query_data):
    """Unpack query data handling both old and new formats."""
    return query_data if len(query_data) == 3 else (*query_data, None)


def process_queries_for_context(agent, query_answer_pairs, dataset_config, metrics, results,
                               query_index, context_index, last_processed_query_id, max_queries,
                               agent_config, output_path, time_cost_list, start_time):
    """Process all queries for a given context."""
    print(f"\n!!!!!Processing {len(query_answer_pairs)} queries for context {context_index}!!!!!\n")
    
    for query_data in tqdm(query_answer_pairs, total=len(query_answer_pairs)):
        query, answer, qa_pair_id = unpack_query_data(query_data)
        
        # Skip queries that have already been processed
        if should_skip_query(query_index, last_processed_query_id):
            logger.info(f"!!!!!Query {query_index} already processed, skipping...\n")
            query_index += 1
            continue
        
        # Check if we've reached the query limit for ablation studies
        if has_reached_query_limit(max_queries, query_index):
            break
        
        # Process the current query
        metrics, results = process_single_query(
            agent, query, answer, dataset_config, metrics, results, query_index, context_index, qa_pair_id
        )
        query_index += 1
        
        # Save results after each query (freq = 1)
        save_results_to_file(output_path, agent_config, dataset_config, results, 
                           metrics, time_cost_list, start_time)
        
    return metrics, results, query_index


def process_context(context_index, context_chunks, query_answer_pairs, agent_config, dataset_config,
                   metrics, results, query_index, last_processed_context_id, last_processed_query_id,
                   max_queries, output_path, time_cost_list, start_time, force_rerun, total_contexts):
    """Process a single context and its queries."""
    # Skip contexts that have already been fully processed
    if should_skip_context(force_rerun, context_index, last_processed_context_id):
        logger.info(f"\n\n!!!!!Experiment {context_index} already finished, skipping...\n")
        return metrics, results, query_index + len(query_answer_pairs), False
    
    # Break early if we've reached the query limit
    if has_reached_query_limit(max_queries, query_index):
        return metrics, results, query_index, True
    
    # Initialize agent for the current context
    agent_save_folder = generate_agent_save_folder(agent_config, dataset_config, context_index)
    agent = initialize_and_memorize_agent(agent_config, dataset_config, agent_save_folder,
                                        context_chunks, context_index, total_contexts)
    
    # Process all queries for this context
    metrics, results, query_index = process_queries_for_context(
        agent, query_answer_pairs, dataset_config, metrics, results,
        query_index, context_index, last_processed_query_id, max_queries,
        agent_config, output_path, time_cost_list, start_time
    )
    
    return metrics, results, query_index, False


def main():
    """Main function to run the memory agent benchmark evaluation."""
    # Parse command line arguments and setup configurations
    args = parse_command_line_arguments()
    agent_config, dataset_config, output_path = setup_configs_and_directories(args)
    
    # Create agent and fetch evaluation data
    start_time, all_context_chunks, all_query_answer_pairs = create_agent_and_fetch_data(
        agent_config, dataset_config
    )
    
    # Load existing results and initialize tracking variables
    time_cost_list = []
    metrics, results, last_processed_context_id, last_processed_query_id = load_existing_results(
        output_path, dataset_config, all_query_answer_pairs, args.force
    )
    
    # Start evaluation loop - process each context and its associated queries
    query_index = 0  # Tracks total queries processed across all contexts
    total_contexts = len(all_context_chunks)
    
    for context_index, (context_chunks, query_answer_pairs) in enumerate(
        tqdm(zip(all_context_chunks, all_query_answer_pairs), total=total_contexts)
    ):
        metrics, results, query_index, should_break = process_context(
            context_index, context_chunks, query_answer_pairs, agent_config, dataset_config,
            metrics, results, query_index, last_processed_context_id, last_processed_query_id,
            args.max_test_queries_ablation, output_path, time_cost_list, start_time,
            args.force, total_contexts
        )
        
        if should_break:
            break
    
    # Log completion
    logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    main()
