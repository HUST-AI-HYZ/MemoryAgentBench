import os
import json
import torch
import tiktoken
from openai import OpenAI
from utils.templates import get_template
from utils.eval_data_utils import (
    format_chat,
)
import re
import time

from langchain_core.documents import Document
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig


class AgentWrapper:
    """
    A wrapper class for different types of memory agents including:
    - Long context agents (GPT, Claude, Gemini)
    - Letta agents
    - Mem0 agents  
    - Cognee agents
    - RAG agents (various implementations)
    """

    def __init__(self, agent_config, dataset_config, load_agent_from):
        """
        Initialize the agent wrapper with specified configuration.
        
        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
            load_agent_from: Optional path to load existing agent state from
        """
        # Basic agent configuration
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        self.context_max_length = dataset_config['context_max_length']
        self.dataset = dataset_config['dataset']
        
        # Output and storage configuration
        self.output_dir = agent_config['output_dir']
        self.agent_save_to_folder = load_agent_from
        
        # Context and token limits
        self.input_length_limit = (agent_config['input_length_limit'] - 
                                 agent_config['buffer_length'] - 
                                 dataset_config['generation_max_length'])
        
        # Model configuration
        self.model = agent_config['model']
        self.max_tokens = dataset_config['generation_max_length']
        self.temperature = agent_config.get('temperature', 0.0)
        
        # Initialize tokenizer (default to gpt-4o-mini for non-gpt models)
        model_for_tokenizer = self.model if "gpt-4o" in self.model else "gpt-4o-mini"
        self.tokenizer = tiktoken.encoding_for_model(model_for_tokenizer)
        
        # Initialize agent based on type
        self._initialize_agent_by_type(agent_config, dataset_config)

    def _initialize_agent_by_type(self, agent_config, dataset_config):
        """Initialize the specific agent type based on agent name."""
        
        if 'Long_context_agent' in self.agent_name:
            self._initialize_long_context_agent()
        elif self._is_agent_type("letta"):
            self._initialize_letta_agent(agent_config, dataset_config)
        elif self._is_agent_type("mem0"):
            self._initialize_mem0_agent(agent_config, dataset_config)
        elif self._is_agent_type("cognee"):
            self._initialize_cognee_agent(agent_config, dataset_config)
        elif self._is_agent_type("zep"):
            self._initialize_zep_agent(agent_config)
        elif self._is_agent_type("amt"):
            self._initialize_amt_agent(agent_config, dataset_config)
        elif self._is_agent_type("rag"):
            self._initialize_rag_agent(agent_config, dataset_config)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _is_agent_type(self, agent_type):
        """Check if the current agent is of a specific type."""
        return agent_type in self.agent_name

    def _create_oai_client(self):
        """Create an OpenAI-compatible client.

        Resolution order:
          1. ``AZURE_OPENAI_ENDPOINT`` + ``AZURE_OPENAI_API_KEY`` (classic Azure OpenAI).
          2. ``AI_FOUNDRY_ENDPOINT`` + ``DefaultAzureCredential`` (mirrors the
             auth model used by AgentMemoryToolkit for embeddings, so a single
             ``az login`` works for both retrieval and chat).
          3. Public OpenAI via ``OPENAI_API_KEY``.

        When using Azure (cases 1 and 2), ``self.model`` must be the Azure
        deployment name, not the model family.
        """
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        ai_foundry_endpoint = os.environ.get("AI_FOUNDRY_ENDPOINT")
        has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
        print(
            f"\n[OAI] resolver: AZURE_OPENAI_ENDPOINT={'set' if azure_endpoint else 'unset'} "
            f"AI_FOUNDRY_ENDPOINT={'set' if ai_foundry_endpoint else 'unset'} "
            f"OPENAI_API_KEY={'set' if has_openai_key else 'unset'}\n"
        )

        # Case 1: classic Azure OpenAI with API key
        if azure_endpoint:
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=azure_endpoint,
            )

        # Case 2: AI Foundry endpoint with Entra ID (DefaultAzureCredential)
        if ai_foundry_endpoint:
            from openai import AzureOpenAI
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            return AzureOpenAI(
                azure_endpoint=ai_foundry_endpoint,
                api_version=os.environ.get(
                    "AZURE_OPENAI_API_VERSION", "2024-10-21"
                ),
                azure_ad_token_provider=token_provider,
            )

        # Case 3: public OpenAI
        if has_openai_key:
            return OpenAI()

        raise RuntimeError(
            "No OpenAI client credentials found. Set one of: "
            "AZURE_OPENAI_ENDPOINT (+ AZURE_OPENAI_API_KEY), "
            "AI_FOUNDRY_ENDPOINT (with DefaultAzureCredential / az login), "
            "or OPENAI_API_KEY."
        )

    def _create_standard_response(self, output, input_tokens, output_tokens, memory_time, query_time):
        """Create standardized response dictionary."""
        return {
            "output": output,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _initialize_long_context_agent(self):
        """Initialize long context agent with appropriate client."""
        self.context = ''
        
        if "gpt" in self.model or "o4" in self.model:
            self.client = self._create_oai_client()
        elif "claude" in self.model:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.environ.get('Anthropic_API_KEY'),
            )
        elif "gemini" in self.model:
            from google import genai
            self.client = genai.Client(api_key=os.environ.get('Google_API_KEY'))
        else:
            raise NotImplementedError(f"Model not supported for long context agent: {self.model}")

    def _initialize_letta_agent(self, agent_config, dataset_config):
        """Initialize Letta agent with proper configuration."""
        if "api" not in agent_config['agent_name']:
            from letta import create_client, LLMConfig, EmbeddingConfig, BasicBlockMemory

            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            
            self.client = create_client()
            self.client.set_default_llm_config(LLMConfig.default_config(agent_config['model']))             
            self.agent_start_time = time.time()
            
            # Configure embedding
            if agent_config['text_embedding'] == 'text-embedding-3-small':
                self.client.set_default_embedding_config(EmbeddingConfig(
                    embedding_model="text-embedding-3-small",
                    embedding_endpoint_type="openai",
                    embedding_endpoint="https://api.openai.com/v1",
                    embedding_dim=1536,
                    embedding_chunk_size=self.chunk_size * 2,
                ))
            else:
                self.client.set_default_embedding_config(
                    EmbeddingConfig.default_config(agent_config['text_embedding'])
                )

            # Load system prompt
            system_path = agent_config['system_path']
            with open(system_path, 'r') as f:
                self.system = f.read()

            # Load or create agent
            if os.path.exists(self.agent_save_to_folder):
                self.load_agent()
            else:
                human_block = self.client.create_block(
                    label='human', 
                    value='User is sharing the contents they are reading recently.', 
                    limit=2000000
                )
                persona_block = self.client.create_block(
                    label='persona', 
                    value='You are a helpful assistant that can help memorize details in the conversation.', 
                    limit=2000000
                )
                memory = BasicBlockMemory(blocks=[human_block, persona_block])
                self.agent_state = self.client.create_agent(
                    name='mm_agent',
                    memory=memory,
                    system=self.system
                )
        ## use the letta api to create the agent
        else:
            from letta_client import Letta, CreateBlock
            
            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            self.agent_start_time = time.time()
            
            
            self.client = Letta(token=os.environ.get('Letta_API_KEY'))
            self.agent_state = self.client.agents.create(
            memory_blocks=[
                CreateBlock(
                    label="human",
                    limit=2000000,
                    value="User is sharing the contents they are reading recently."
                ),
                CreateBlock(
                    label="persona",
                    limit=2000000,
                    value="You are a helpful assistant that can help memorize details in the conversation."
                )
            ],
            model=f"openai/{agent_config['model']}",
            embedding=f"openai/{agent_config['text_embedding']}"
        )

            
            
    def _initialize_mem0_agent(self, agent_config, dataset_config):
        """Initialize Mem0 agent with retrieval configuration."""
        from mem0.memory.main import Memory
        
        self.retrieve_num = agent_config['retrieve_num']
        self.context = ''
        self.client = self._create_oai_client()
        self.memory = Memory()
        self.agent_start_time = time.time()

    def _initialize_amt_agent(self, agent_config, dataset_config):
        """Initialize AgentMemoryToolkit (Cosmos DB) agent."""
        from agent_memory_toolkit import CosmosMemoryClient

        self.retrieve_num = agent_config['retrieve_num']
        self.context = ''
        self.client = self._create_oai_client()

        cosmos_kwargs = {
            "cosmos_endpoint": os.environ.get("COSMOS_DB_ENDPOINT"),
            "cosmos_database": os.environ.get("COSMOS_DB_DATABASE"),
            "cosmos_container": os.environ.get("COSMOS_DB_CONTAINER"),
            "cosmos_counter_container": os.environ.get("COSMOS_DB_COUNTERS_CONTAINER", "counter"),
            "cosmos_lease_container": os.environ.get("COSMOS_DB_LEASE_CONTAINER", "leases"),
            "cosmos_throughput_mode": os.environ.get("COSMOS_DB_THROUGHPUT_MODE", "serverless"),
            "cosmos_autoscale_max_ru": int(os.environ.get("COSMOS_DB_AUTOSCALE_MAX_RU", "1000")),
            "ai_foundry_endpoint": os.environ.get("AI_FOUNDRY_ENDPOINT"),
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large"),
            "use_default_credential": True,
        }
        adf_endpoint = os.environ.get("ADF_ENDPOINT")
        if adf_endpoint:
            cosmos_kwargs["adf_endpoint"] = adf_endpoint
            cosmos_kwargs["adf_key"] = os.environ.get("ADF_KEY", "")

        self.memory = CosmosMemoryClient(**cosmos_kwargs)
        try:
            self.memory.create_memory_store()
        except Exception as exc:
            print(f"\n[AMT] create_memory_store skipped: {exc}\n")

        # Optional agentic-memory post-processing run after all chunks
        # are memorized. One of: "none" (default), "facts", "summary", "both".
        self.amt_processing_mode = agent_config.get('amt_processing_mode', 'none')
        self._amt_user_id = None
        self._amt_thread_id = None
        self.agent_start_time = time.time()

    def _initialize_cognee_agent(self, agent_config, dataset_config):
        """Initialize Cognee agent with knowledge graph configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.agent_start_time = time.time()
        self.cognee_dir = './cognee/.cognee_system/databases/cognee.lancedb'
    
    def _initialize_zep_agent(self, agent_config):
        # from zep_cloud.client import AsyncZep
        # self.client = AsyncZep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.development.getzep.com/api/v2")
        from zep_cloud import Zep
        from methods.zep import OpenAIAgent
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context_id = -1

        self.client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.oai_client = OpenAIAgent(model=self.model, source="azure", api_dict={"endpoint":os.environ.get("AZURE_OPENAI_ENDPOINT"), "api_version":os.environ.get("AZURE_OPENAI_API_VERSION"), "api_key":os.environ.get("AZURE_OPENAI_API_KEY")}, temperature=self.temperature)
        self.agent_start_time = time.time()

    def _initialize_rag_agent(self, agent_config, dataset_config):
        """Initialize RAG agent with retrieval configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = dataset_config['chunk_size']
        self.context_len = 0
        self.context_id = -1

    def send_message(self, message, memorizing=False, query_id=None, context_id=None):
        """
        Send a message to the agent for either memorization or querying.
        
        Args:
            message: The message content (context for memorization, query for answering)
            memorizing: Whether to memorize the message (True) or answer it (False)
            query_id: Unique identifier for the query
            context_id: Unique identifier for the context
            
        Returns:
            dict or str: Agent response with metadata (for queries) or confirmation (for memorization)
        """
        # Route to appropriate agent handler based on agent type
        if 'Long_context_agent' in self.agent_name:
            return self._handle_long_context_agent(message, memorizing)
        elif any(self._is_agent_type(agent_type) for agent_type in ["letta", "cognee", "mem0", "zep", "amt"]):
            return self._handle_memory_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("rag"):
            return self._handle_rag_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _handle_long_context_agent(self, message, memorizing):
        """Handle message processing for long context agents."""
        if memorizing:
            # Add message to context memory
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            # Process query with context
            return self._query_long_context_agent(message)

    def _query_long_context_agent(self, message):
        """Process a query for long context agents."""
        # Get appropriate tokenizer
        try:
            tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Handle context truncation for non-long context models
        buffer_length = 50000
        if self.input_length_limit <= self.context_max_length + buffer_length:
            self._truncate_context_if_needed(tokenizer)
                
        # Format message with context and system prompt
        full_message = self.context + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        formatted_message = format_chat(message=full_message, system_message=system_message)
        
        # Query the model
        start_time = time.time()
        
        if "gpt" in self.model: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return self._format_openai_response(response, start_time)
            
        elif "o4" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
            )
            return self._format_openai_response(response, start_time)
            
        elif "claude" in self.model:
            return self._query_claude(full_message, system_message, start_time)
            
        elif "gemini" in self.model:
            return self._query_gemini(formatted_message, start_time)
            
        else:
            raise NotImplementedError(f"Model not supported: {self.model}")

    def _truncate_context_if_needed(self, tokenizer):
        """Truncate context if it exceeds limits."""
        # Truncate context if it exceeds the context_max_length
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.context_max_length:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.context_max_length:])
        
        # Truncate if context exceeds the input_length_limit
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])

    def _format_openai_response(self, response, start_time):
        """Format OpenAI API response into standard output format."""
        return self._create_standard_response(
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            0,
            time.time() - start_time
        )

    def _query_claude(self, message, system_message, start_time):
        """Query Claude model with proper formatting."""
        formatted_message = format_chat(message=message, system_message=system_message, include_system=False)
        response = self.client.messages.create(
            model=self.model,
            messages=formatted_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._create_standard_response(
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
            0,
            time.time() - start_time
        )

    def _query_gemini(self, formatted_message, start_time):
        """Query Gemini model with proper configuration."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=formatted_message[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=formatted_message[0]["content"], 
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )      
        )
        return self._create_standard_response(
            response.text,
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count,
            0,
            time.time() - start_time
        )
        
    def _handle_memory_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for memory-based agents (Letta, Cognee, Mem0)."""
        if self._is_agent_type("letta"):
            return self._handle_letta_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("cognee"):
            return self._handle_cognee_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("mem0"):
            return self._handle_mem0_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("zep"):
            return self._handle_zep_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("amt"):
            return self._handle_amt_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Memory agent type not supported: {self.agent_name}")

    def _handle_letta_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Letta agents."""
        # Format message based on context
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
        else:
            formatted_message = message
        
        # Handle memory construction time for queries
        memory_construction_time = 0 if memorizing else time.time() - self.agent_start_time
        
        # Reload agent for queries
        if not memorizing:
            if os.path.exists(self.agent_save_to_folder):
                self.load_agent()
            else:
                print(f"\n\nAgent {self.agent_name} not found in {self.agent_save_to_folder}\n\n")
        
        # Process based on Letta mode
        response = self._process_letta_message(formatted_message, memorizing, query_id, context_id)
        
        if memorizing:
            return "Memorized"
        
        # Create response for queries
        tokenizer = self.tokenizer
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            len(tokenizer.encode(message, disallowed_special=())),
            len(tokenizer.encode(response, disallowed_special=())),
            memory_construction_time,
            query_time_len
        )
        self.agent_start_time = time.time()  # Reset time
        return output
    
    def _process_letta_message(self, formatted_message, memorizing, query_id, context_id):
        """Process message with Letta client based on mode."""
        from letta_client import Letta, MessageCreate
        
        try:
            if self.letta_mode == 'insert':
                if memorizing:
                    self.client.server.passage_manager.insert_passage(
                        agent_state=self.agent_state,
                        agent_id=self.agent_state.id,
                        text=formatted_message,
                        actor=self.client.user,
                    )
                    # import ipdb; ipdb.set_trace()
                    return "Memorized"
                else:
                    response = self.client.send_message(
                        agent_id=self.agent_state.id,
                        message=formatted_message,
                        role='user')
                    ## save response.messages to a file / for debugging as JSON     
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
            
            elif self.letta_mode == 'chat':
                response = self.client.send_message(
                    agent_id=self.agent_state.id,
                    message=formatted_message,
                    role='user')
                
                if memorizing:
                    return "Memorized"
                else:
                    ## save response.messages to a file / for debugging as JSON    
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
            elif self.letta_mode == 'api':
                response = self.client.agents.messages.create(
                    agent_id=self.agent_state.id,
                    messages=[
                        MessageCreate(
                            role="user",
                            content=formatted_message,
                        ),
                    ],
                )
                print(f"\n\n\nresponse: {response}\n\n\n")
                return response.messages[-1].content
        except Exception as e:
            print(f"\n\n\nerror: {e}\n\n\n")
            return "Error"

    def _handle_cognee_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Cognee agents."""
        import cognee
        import asyncio
        dataset_name = f'default_dataset_{self.sub_dataset}_context_{context_id}'
        
        if memorizing:
            # Add context to Cognee knowledge base
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            
            # Add text to cognee and generate knowledge graph
            asyncio.run(cognee.add(formatted_message, dataset_name=dataset_name))
            asyncio.run(cognee.cognify(datasets=[dataset_name], chunk_size=self.chunk_size))

            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:                    
            # Query the knowledge graph
            memory_construction_time = time.time() - self.agent_start_time
            searched_results = asyncio.run(cognee.search(
                query_text=message, 
                top_k=self.retrieve_num, 
                datasets=[dataset_name]
            ))
                    
            # Format results
            total_results = ("".join([f"{result}\n" for result in searched_results]) 
                           if searched_results else "No results found.")
            
            # Return formatted output
            tokenizer = self.tokenizer
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            output = self._create_standard_response(
                total_results,
                len(tokenizer.encode(self.context, disallowed_special=())),
                len(tokenizer.encode(total_results, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            return output

    def _handle_mem0_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Mem0 agents."""
        user_id = f'context_{context_id}_{self.sub_dataset}'
        if memorizing:
            system_message = get_template(self.sub_dataset, 'system', self.agent_name)
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            
            # Generate Assistant response
            # memory_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": formatted_message}]
            # response = OpenAI().chat.completions.create(
            #             model=self.model,
            #             messages=memory_messages,
            #             max_tokens=1000,
            #         )
            # memory_messages = [
            #     {"role": "system", "content": system_message}, 
            #     {"role": "user", "content": formatted_message},
            #     {"role": "assistant", "content": response.choices[0].message.content}
            # ]
            memory_messages = [
                {"role": "system", "content": system_message}, 
                {"role": "user", "content": formatted_message},
                {"role": "assistant", "content": "I'll make sure to add the content into the memory."}
            ]
            
            vector_results = self.memory.add(memory_messages, user_id=user_id)
            print(f"\n\n\nvector_results: {vector_results}\n\n\n")
            return "Memorized"
        else:
            # Retrieve relevant memories and generate response
            memory_construction_time = time.time() - self.agent_start_time
            relevant_memories = self.memory.search(query=message, user_id=user_id, limit=self.retrieve_num)
            print(f"\n\n\nrelevant_memories: {relevant_memories}\n\n\n")
            
            memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
            
            # Generate assistant response
            system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\n{memories_str}\n"
            llm_messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": message + "\n\nCurrent Time: " + time.strftime("%Y-%m-%d %H:%M:%S")}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=llm_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            memory_retrieval_length = len(self.tokenizer.encode(memories_str, disallowed_special=()))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            print(f"\nmemory_length: {memory_retrieval_length}\n")
            
            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens + memory_retrieval_length,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            return output
    
    # AgentMemoryToolkit (Cosmos DB)
    def _handle_amt_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for AgentMemoryToolkit (Cosmos DB) agents."""
        # Derive a stable user/thread id from the per-context save folder so
        # memorize (context_id=None) and query (context_id=int) end up under
        # the same partition. Falls back to context_id if no folder is set.
        folder_tag = os.path.basename(self.agent_save_to_folder.rstrip('/\\')) \
            if self.agent_save_to_folder else f'ctx{context_id}'
        user_id = f'amt_{self.sub_dataset}_{folder_tag}'
        thread_id = user_id
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(
                context=message,
                **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {})
            )
            self.memory.add_cosmos(
                user_id=user_id,
                role="user",
                content=formatted_message,
                thread_id=thread_id,
            )
            self._amt_user_id = user_id
            self._amt_thread_id = thread_id
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time
            relevant_memories = self.memory.search_cosmos(
                search_terms=message,
                user_id=user_id,
                top_k=self.retrieve_num,
            )
            print(f"\n\n\nrelevant_memories count: {len(relevant_memories)}\n\n\n")

            # Truncate memories to fit within input_length_limit (token budget)
            entries = [f"- {entry['content']}" for entry in relevant_memories]
            kept = []
            running_tokens = 0
            for entry in entries:
                entry_tokens = len(self.tokenizer.encode(entry, disallowed_special=()))
                if running_tokens + entry_tokens > self.input_length_limit:
                    break
                kept.append(entry)
                running_tokens += entry_tokens
            if len(kept) < len(entries):
                print(f"\n[AMT] truncated retrieved memories: kept {len(kept)}/{len(entries)} "
                      f"(~{running_tokens} tokens, limit {self.input_length_limit})\n")
            memories_str = "\n".join(kept)
            
            system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\n{memories_str}\n"
            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message + "\n\nCurrent Time: " + time.strftime("%Y-%m-%d %H:%M:%S")}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=llm_messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens
            )
            
            memory_retrieval_length = len(self.tokenizer.encode(memories_str, disallowed_special=()))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            print(f"\nmemory_length: {memory_retrieval_length}\n")
            
            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens + memory_retrieval_length,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()
            return output

    # Zep
    def _handle_zep_agent(self, message, memorizing, query_id, context_id):
        """Handle Zep processing."""
        import inspect
        from zep_cloud import Message
        from methods.zep import compose_search_context, llm_response, get_retrieval_query, construct_messages
        
        # user id / session id / oai client
        user_id = f'user_{context_id}_{self.sub_dataset}'
        graph_id = f'graph_{context_id}_{self.sub_dataset}'
        thread_id = f'thread_{context_id}_{self.sub_dataset}'
                
        # check the context id for user and session creation
        if self.context_id != context_id and memorizing:
            # User creation
            self.client.user.add(user_id=user_id)
            
            # Thread creation
            self.client.thread.create(thread_id=thread_id, user_id=user_id)
                    
            # Graph creation
            self.client.graph.create(graph_id=graph_id)
            self.context_id = context_id
        else:
            pass
            
        if memorizing:
            # graph add
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            content = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.client.graph.add(
                graph_id=graph_id, 
                type="text",
                data=content[:9998]
            )

            # # thread add
            messages = construct_messages(content, user_id)
            self.client.thread.add_messages(thread_id=thread_id, messages=messages)
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time
            
            # graph search
            retrieval_query = get_retrieval_query(message)
            print(f"\n\n\nretrieval_query: {retrieval_query}\n\n\n")

            edges_results = self.client.graph.search(graph_id=graph_id, query=retrieval_query[:399], scope='edges', limit=self.retrieve_num).edges
            node_results = self.client.graph.search(graph_id=graph_id, query=retrieval_query[:399], scope='nodes', limit=self.retrieve_num).nodes
            episode_results = self.client.graph.search(graph_id=graph_id, query=retrieval_query[:399], scope='episodes', limit=self.retrieve_num).episodes
            
            # print(f"\n\n\nepisode_results: {episode_results}\n\n\n")
            # print(f"\n\n\nedges_results: {edges_results}\n\n\n")
            # print(f"\n\n\nnode_results: {node_results}\n\n\n")
                        
            # thread search / currently we do not use the thread info
            memory = self.client.thread.get_user_context(thread_id=thread_id)
            context_block = memory.context

            # Prompt an LLM with relevant context
            retrieved_context = compose_search_context(edges_results, node_results, context_block, episode_results)
            import asyncio
            response = asyncio.run(llm_response(self.oai_client, retrieved_context, message))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time

            output = self._create_standard_response(
                response,
                len(self.tokenizer.encode(retrieved_context, disallowed_special=())),
                len(self.tokenizer.encode(response, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            
            # save the context
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w") as f:
                paragraphs = [p for p in retrieved_context.replace("\r\n", "\n").split("\n") if p.strip()]
                json.dump({"retrieved_context_paragraphs": paragraphs, "response": response}, f, ensure_ascii=False, indent=2)
            
            return output
    
    def _handle_rag_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for RAG agents."""
        if memorizing:
            # Add message to chunks and context
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            self.chunks.append(formatted_message)
            self.context_len = self.context_len + self.chunk_size
            
            # Truncate context if it exceeds limits
            if self.context_len > self.input_length_limit:
                self.chunks = self.chunks[1:]
                self.context_len = self.context_len - self.chunk_size
            return ''
        else:
            # Handle query processing for different RAG types
            return self._process_rag_query(message, query_id, context_id)

    def _process_rag_query(self, message, query_id, context_id):
        """Process query for RAG agents with different retrieval strategies."""
                
        # Truncate context if needed
        tokenizer = self.tokenizer
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])
        if self.context_len > self.input_length_limit:
            self.chunks = self.chunks[1:]
            self.context_len = self.context_len - self.chunk_size
        
        # Route to specific RAG implementation and get result
        rag_handlers = {
            "graph_rag": lambda: self._handle_graph_rag(message, context_id, tokenizer),
            "hippo_rag_v2_nv": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "hippo_rag_v2_openai": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "rag_bm25": lambda: self._handle_bm25_rag(message, context_id, tokenizer),
            "rag_contriever": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_large": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_small": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_qwen3_embedding_4b": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_raptor": lambda: self._handle_raptor_rag(message, context_id, tokenizer),
            "self_rag": lambda: self._handle_self_rag(message, context_id, tokenizer),
            "memo_rag": lambda: self._handle_memorag(message, context_id, tokenizer),
        }
        
        # Find matching handler
        handler = next((handler for agent_type, handler in rag_handlers.items() if self._is_agent_type(agent_type)), None)
        if not handler:
            raise NotImplementedError(f"RAG agent type not supported: {self.agent_name}")
        
        output = handler()

        # Save the retrieved context as JSON (if the method provides it)
        if output.get("retrieval_context"):
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w") as f:
                json.dump(output["retrieval_context"], f)
            
            # drop the retrieval_context       
            output.pop("retrieval_context")
        
        return output

    def _handle_graph_rag(self, message, context_id, tokenizer):
        """Handle Graph RAG processing."""
        start_time = time.time()

        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            try:
                from methods.graph_rag import GraphRAG
                self.graph_rag = GraphRAG(temperature=self.temperature, model_name=self.model, retrieve_num=self.retrieve_num, max_tokens=self.max_tokens)
                self.graph_rag.process_documents(docs)
                memory_construction_time = time.time() - start_time
            except Exception as e:
                print(f"\n\n\n\nError: {e}\n\n\n\n")
            print(f"\n\nGraph RAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping Graph RAG build vectorstore...\n\n")

        # Process query
        try:
            response, retrieval_context = self.graph_rag.query(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context = "ERROR"
            print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        self.context_id = context_id
        
        print(f"\n\n\n\nResponse: {response}\n\n\n\n")
        if isinstance(response, str):
            response = response
        else:
            response = response.content
        query_time_len = time.time() - start_time - memory_construction_time
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    def _handle_hippo_rag(self, message, context_id, tokenizer):
        """Handle HippoRAG processing."""
        start_time = time.time()
        
        if self.context_id != context_id:
            docs = self.chunks
            from methods.hipporag import HippoRAG
            if any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_nv"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/NV-Embed-v2", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}')
                embedding_model_name = 'nvidia/NV-Embed-v2'
            elif any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_openai"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/OpenAIEmbedding", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}') 
                embedding_model_name = 'text-embedding-ada-002'
            
            self.hipporag = HippoRAG(save_dir=save_dir,
                                llm_model_name=self.model,
                                embedding_model_name=embedding_model_name) 
            self.hipporag.index(docs=docs)
            memory_construction_time = time.time() - start_time
            print(f"\n\nHippoRAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping HippoRAG build vectorstore...\n\n")
            
        # Retrieve and answer
        queries = [message]
        retrieval_results, top_k_docs = self.hipporag.retrieve(queries=queries, num_to_retrieve=self.retrieve_num)
        
        qa_results = self.hipporag.rag_qa(retrieval_results)
        response = qa_results[0][0].answer
        
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(top_k_docs)])
        query_time_len = time.time() - start_time - memory_construction_time
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    # RAG implementation methods
    def _handle_bm25_rag(self, message, context_id, tokenizer):
        """Handle BM25 RAG processing."""
        start_time = time.time()
        
        # Extract retrieval query from message
        retrieval_query = self._extract_retrieval_query(message)
        print(f"\n\n\n\nretrieval_query: {retrieval_query}\n\n\n\n")
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            from langchain_community.retrievers import BM25Retriever
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            print(f"\n\nBM25 build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping BM25 build vectorstore...\n\n")
        
        # Retrieve documents
        self.bm25_retriever.k = self.retrieve_num
        bm25_documents = self.bm25_retriever.get_relevant_documents(retrieval_query)   
        retrieval_context = [f"{doc.page_content}\n" for doc in bm25_documents] 
        memory_construction_time = time.time() - start_time
        
        # Answer the query
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieval_context)])
        
        # Format the message
        ask_llm_message = retrieval_memory_string + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        format_message = format_chat(message=ask_llm_message, system_message=system_message)
        
        # Generate response
        response = self._create_oai_client().chat.completions.create(
            model=self.model,
            messages=format_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens if "gpt-4" in self.model else None
        )
        
        query_time_len = time.time() - start_time - memory_construction_time
        self.context_id = context_id
        
        return {
            "output": response.choices[0].message.content,
            "input_len": response.usage.prompt_tokens,
            "output_len": response.usage.completion_tokens,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
    
    def _extract_retrieval_query(self, message):
        """Extract retrieval query from message using regex patterns."""
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return ''.join(match.groups())
        
        return message
        
    def _handle_embedding_rag(self, message, context_id, tokenizer):
        """Handle embedding-based RAG processing (Contriever, Text-embedding models)."""
        from methods.embedding_retriever import TextRetriever, RAGSystem
        
        # Determine embedding model
        if any(agent_name in self.agent_name for agent_name in ["rag_contriever"]):
            embedding_model_name = "facebook/contriever"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_large"]):
            embedding_model_name = "text-embedding-3-large"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_small"]):
            embedding_model_name = "text-embedding-3-small"
        elif any(agent_name in self.agent_name for agent_name in ["rag_qwen3_embedding_4b"]):
            embedding_model_name = "Qwen/Qwen3-Embedding-4B"
        else:
            raise NotImplementedError
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            self.retriever = TextRetriever(embedding_model_name=embedding_model_name)
            self.retriever.build_vectorstore(self.chunks)
            print(f"\n\n{embedding_model_name} build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping {embedding_model_name} build vectorstore...\n\n")
                            
        # Retrieve relevant passages and answer the query
        rag_system = RAGSystem(self.retriever, self.model, self.temperature, self.max_tokens, use_azure=True, azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"), azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"), azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"))
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        result = rag_system.answer_query(
            query=message, 
            top_k=self.retrieve_num, 
            system_message=system_message
        )
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": result["answer"],
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(result["answer"], disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_raptor_rag(self, message, context_id, tokenizer):
        """Handle RAPTOR RAG processing."""
        # Build vectorstore if context changed
        if self.context_id != context_id:
            texts = self.chunks
            from methods.raptor import RAPTORMethod
            self.raptor_method = RAPTORMethod(texts, max_levels=3)
            print(f"\n\nRaptor build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Raptor build vectorstore...\n\n")
        
        # Retrieve relevant passages and answer the query
        result = self.raptor_method.run(query=message, k=self.retrieve_num)
        response = result['answer']
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_self_rag(self, message, context_id, tokenizer):
        """Handle Self-RAG processing."""
        from methods.self_rag import SelfRAG
        start_time = time.time()
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.self_rag = SelfRAG(documents=docs, temperature=self.temperature, top_k=self.retrieve_num)
            print(f"\n\nSelf-RAG build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Self-RAG build vectorstore...\n\n")
        
        # Process query
        try:
            response, retrieval_context_list, memory_construction_time, query_time_len = self.self_rag.run(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context_list = ["ERROR"]
            memory_construction_time = 0
            query_time_len = 0
            print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        # Prepare the context
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" 
                                        for i, text in enumerate(retrieval_context_list)])
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    # memorag
    def _handle_memorag(self, message, context_id, tokenizer):
        """Handle MemoRAG processing."""
        from methods.memorag import Agent, MemoRAG
        start_time = time.time()
        memory_construction_time = 0
        cache_context_save_dir=f"./outputs/rag_retrieved/MemoRAG/{self.sub_dataset}/chunksize_{self.chunk_size}/context_id_{context_id}"
        
        # build rag agent
        if self.context_id != context_id:
            # API configuration
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
            api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            gen_model = Agent(model=self.model, source="azure", temperature=self.temperature, api_dict={"endpoint":endpoint, "api_version":api_version, "api_key":api_key})
            self.MemoRAG = MemoRAG(
                mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
                ret_model_name_or_path="BAAI/bge-m3",   
                customized_gen_model=gen_model,
                ret_hit=self.retrieve_num, 
                retrieval_chunk_size=self.chunk_size
            )
            # Use the loaded context / memorize the context for question answering
            context = " ".join(self.chunks)
            ## load the context from the cache
            if os.path.exists(f'{cache_context_save_dir}/memory.bin'):
                self.MemoRAG.load(cache_context_save_dir, print_stats=True)
            else:
                self.MemoRAG.memorize(context, save_dir=None, print_stats=True)
            memory_construction_time = time.time() - start_time
            print(f"Finish memorizing, time cost {memory_construction_time}")
        else:
            print(f"\n\nContext {context_id} already processed, skipping MemoRAG build vectorstore...\n\n")
            
        # Retrieve and answer
        if self.sub_dataset == "infbench_sum_eng_shots2":
            response, retrieval_context = self.MemoRAG(query=message, task_type="summarize", max_new_tokens=self.max_tokens)
        else:
            response, retrieval_context = self.MemoRAG(query=message, task_type="memorag", max_new_tokens=self.max_tokens)
        
        query_time_len = time.time() - start_time - memory_construction_time
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(str(retrieval_context) + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
        
    def _run_amt_post_processing(self):
        """Trigger AMT Durable Function pipelines after memorize phase.

        Controlled by ``amt_processing_mode`` in the agent YAML:
          - ``none``    : skip (default)
          - ``facts``   : extract_facts only
          - ``summary`` : generate_thread_summary only
          - ``both``    : both pipelines
        Outputs land in the same Cosmos container as turns and are
        therefore picked up by subsequent ``search_cosmos`` queries.
        """
        mode = getattr(self, 'amt_processing_mode', 'none')
        if mode == 'none' or not self._amt_user_id or not self._amt_thread_id:
            return
        try:
            if mode in ('facts', 'both'):
                print(f"\n[AMT] extract_facts user={self._amt_user_id} thread={self._amt_thread_id}\n")
                self.memory.extract_facts(user_id=self._amt_user_id, thread_id=self._amt_thread_id)
            if mode in ('summary', 'both'):
                print(f"\n[AMT] generate_thread_summary user={self._amt_user_id} thread={self._amt_thread_id}\n")
                self.memory.generate_thread_summary(user_id=self._amt_user_id, thread_id=self._amt_thread_id)
        except Exception as exc:
            print(f"\n[AMT] post-processing failed: {exc}\n")

    def save_agent(self):
        """Save agent state to disk for persistence."""
        # AMT: trigger optional Durable Functions post-processing after memorize
        if self._is_agent_type("amt"):
            self._run_amt_post_processing()
            # Drop a marker so future runs skip re-memorization
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/amt_marker.txt", "w") as f:
                f.write(f"user_id={self._amt_user_id}\nthread_id={self._amt_thread_id}\n")
            print("\n\n Agent saved (amt: cosmos-backed) \n\n")
            return

        # Currently only implemented for Letta agents
        if not self._is_agent_type("letta") and not self._is_agent_type("zep"):
            print("\n\n Agent not saved (not implemented for this agent type) \n\n")
            return
        
        if self._is_agent_type("letta") and "api" not in self.agent_name:
            agent_save_folder = self.agent_save_to_folder
            os.makedirs(agent_save_folder, exist_ok=True)
            
            import shutil
            # Copy the SQLite database file to the target folder
            source_db_path = os.path.expanduser("~/.letta/sqlite.db")
            target_db_path = f"{agent_save_folder}/sqlite.db"
            shutil.copyfile(source_db_path, target_db_path)
            
            # Save the agent ID for future loading
            with open(f"{agent_save_folder}/agent_id.txt", "w") as f:
                f.write(self.agent_state.id)
        elif self._is_agent_type("zep"):
            # save the message that agent has processed
            messages = "agent finished memorization"
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "w") as f:
                f.write(messages)
                
        print("\n\n Agent saved...\n\n")

    def load_agent(self):
        """Load agent state from disk."""
        agent_save_folder = self.agent_save_to_folder
        assert os.path.exists(agent_save_folder), f"Folder {agent_save_folder} does not exist."

        if not self._is_agent_type("letta") and not self._is_agent_type("zep") and not self._is_agent_type("amt"):
            print("\n\nAgent loading not implemented for this agent type\n\n")
            return None

        if self._is_agent_type("letta") and "api" not in self.agent_name:
            import shutil
            # Copy the database file back to the Letta directory
            source_db_path = f"{agent_save_folder}/sqlite.db"
            target_db_path = os.path.expanduser("~/.letta/sqlite.db")
            shutil.copyfile(source_db_path, target_db_path)

            # Load agent ID and find the corresponding agent state
            with open(f"{agent_save_folder}/agent_id.txt", "r") as f:
                agent_id = f.read()

            # Find the agent state with the matching ID
            for agent_state in self.client.list_agents():
                if agent_state.id == agent_id:
                    self.agent_state = agent_state
                    break
        elif self._is_agent_type("zep"):
            # load the message that agent has processed
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "r") as f:
                messages = f.read()
        
        print("\n\n Agent loaded successfully...\n\n")
        