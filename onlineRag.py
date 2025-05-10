# Flowchart: https://claude.ai/public/artifacts/af9c4502-17d8-4300-9cef-e03dbcfb86fd

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime

# External libraries
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

"""
    A RAG system that intelligently decides when to make external network requests
    based on query analysis, retrieval quality, and time sensitivity.
"""
class SmartRAG:
    def __init__(self, 
                 llama_index_name: str = "Dandesign-17159", 
                 llama_project_name: str = "Default",
                 gpt_model: str = "gpt-4",
                 perplexity_model: str = "sonar-pro",
                 retriever_k: int = 5,
                 min_relevance_threshold: float = 0.7,
                 time_sensitivity_threshold: float = 0.8,
                 enable_logging: bool = True):
        """
        Initialize the SmartRAG system.
        
        Args:
            gpt_model: OpenAI model to use for relevance scoring and decision making
            perplexity_model: Perplexity model to use for external requests
            retriever_k: Number of documents to retrieve for each query
            min_relevance_threshold: Minimum relevance score to avoid external request
            time_sensitivity_threshold: Threshold for determining time-sensitive queries
            enable_logging: Whether to log decisions and scores
        """
        
        self.enable_logging = enable_logging
        self.retriever_k = retriever_k
        self.min_relevance_threshold = min_relevance_threshold
        self.time_sensitivity_threshold = time_sensitivity_threshold
        
        # Initialize LlamaIndex retriever
        self.index = LlamaCloudIndex(
            name=llama_index_name,
            project_name=llama_project_name,
            organization_id=os.getenv("LLAMA_INDEX_ORG_ID"),
            api_key=os.getenv("LLAMA_INDEX_KEY"),
        )
        
        # Configure retriever with hybrid search
        self.retriever = self.index.as_retriever(
            dense_similarity_top_k=retriever_k,
            sparse_similarity_top_k=retriever_k,
            alpha=0.5,
            enable_reranking=True,
            include_metadata=True
        )
        
        # Initialize GPT for relevance scoring and decision making
        self.gpt = ChatOpenAI(model=gpt_model)
        
        # Initialize Perplexity for external network requests
        self.perplexity = ChatPerplexity(temperature=0.7, model=perplexity_model)
        
        # Set up prompts
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Set up the prompts for different components of the system."""
        
        # Prompt for relevance scoring
        self.relevance_scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at evaluating the relevance of retrieved passages to a user query.
                Task: Analyze the passages below and determine how well they address the user's query.
                Output a JSON with:
                1. relevance_score: A float from 0.0-1.0 representing overall relevance 
                2. intent_coverage: A boolean indicating if the passages truly address the core intent
                3. reasoning: A brief explanation of your evaluation
                4. domain_authority_score: A float from 0.0-1.0 representing the authority of the sources

                Higher scores should be given to passages from:
                - Official sources (government, company websites)
                - Reputable media outlets
                - Academic journals
                Lower scores should be given to:
                - Forums
                - Personal blogs
                - Social media
            """),
            ("human", """Query: {query}
            
Retrieved Passages:
{passages}

Evaluate the relevance and respond in JSON format only.
            """)
        ])
        
        # Prompt for time sensitivity detection
        self.time_sensitivity_prompt = ChatPromptTemplate.from_messages([
            ("system", 
"""You analyze user queries to determine if they require time-sensitive, up-to-date information.
            
    Task: Analyze the query and output a JSON with:
    1. time_sensitive: A float from 0.0-1.0 representing how time-sensitive the query is
    2. reasoning: A brief explanation for your rating
    3. recency_required: A string indicating the recency needed ("day", "week", "month", "year", or "none")

    High time sensitivity (0.8-1.0) applies to queries that:
    - Explicitly mention current events, news, or recent developments
    - Use terms like "latest", "current", "today", "this week"
    - Ask about evolving situations or ongoing events
    - Reference very recent dates
            """),
            ("human", "Query: {query}")
        ])
        
        # Prompt for making the final decision
        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You decide whether to use internal knowledge base results or make an external network request.
        
Task: Review the query analysis below and decide if an external network request is needed.
Output a JSON with:
1. make_external_request: A boolean indicating whether to make an external request
2. reasoning: A brief explanation for your decision
3. confidence: A float from 0.0-1.0 representing confidence in this decision
            """),
            ("human", """Query: {query}
            
Relevance Analysis: {relevance_analysis}
Time Sensitivity Analysis: {time_sensitivity_analysis}

Based on this information, decide whether to use the internal knowledge base or make an external network request.
            """)
        ])
        
        # Prompt for external requests via Perplexity
        self.perplexity_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to up-to-date information.
            Use your knowledge to answer the user's query accurately and comprehensively.
            Cite your sources where appropriate.
            """),
            ("human", "{query}")
        ])
    
    def evaluate_passage_relevance(self, query: str, passages: List[str]) -> Dict[str, Any]:
        """
        Evaluate the relevance of retrieved passages to the query.
        
        Args:
            query: The user's query
            passages: Retrieved passages from the knowledge base
            
        Returns:
            Dict containing relevance scores and analysis
        """
        formatted_passages = "\n\n".join(f"Passage {i+1}:\n{passage}" for i, passage in enumerate(passages))
        
        response = self.gpt.invoke(
            self.relevance_scoring_prompt.format(
                query=query,
                passages=formatted_passages
            )
        )
        
        # Parse the JSON response
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback in case of parsing error
            return {
                "relevance_score": 0.5,
                "intent_coverage": False,
                "reasoning": "Failed to parse response",
                "domain_authority_score": 0.5
            }
    
    def evaluate_time_sensitivity(self, query: str) -> Dict[str, Any]:
        """
        Evaluate the time sensitivity of the query.
        
        Args:
            query: The user's query
            
        Returns:
            Dict containing time sensitivity analysis
        """
        response = self.gpt.invoke(
            self.time_sensitivity_prompt.format(query=query)
        )
        
        # Parse the JSON response
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback in case of parsing error
            return {
                "time_sensitive": 0.3,
                "reasoning": "Failed to parse response",
                "recency_required": "none"
            }
    
    def make_decision(self, query: str, relevance_analysis: Dict[str, Any], 
                     time_sensitivity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to make an external network request.
        
        Args:
            query: The user's query
            relevance_analysis: Results from evaluate_passage_relevance
            time_sensitivity_analysis: Results from evaluate_time_sensitivity
            
        Returns:
            Dict containing the decision and reasoning
        """
        response = self.gpt.invoke(
            self.decision_prompt.format(
                query=query,
                relevance_analysis=json.dumps(relevance_analysis),
                time_sensitivity_analysis=json.dumps(time_sensitivity_analysis)
            )
        )
        
        # Parse the JSON response
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback in case of parsing error
            return {
                "make_external_request": True,
                "reasoning": "Failed to parse decision response, defaulting to external request",
                "confidence": 0.5
            }
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Main query method that orchestrates the entire process.
        
        Args:
            user_query: The user's query
            
        Returns:
            Dict containing the response and metadata
        """
        # Step 1: Retrieve passages from internal knowledge base
        
        """
        ==========================Need to be modified===============================
        * Comment the following two lines:
            nodes = self.retriever.retrieve(user_query)
            passages = [node.text for node in nodes]
            
        * What you need to do:
            Call your function, to return all the fragments from RAG as a LIST -> passages = [node.text for node in nodes]
        """
        nodes = self.retriever.retrieve(user_query)
        passages = [node.text for node in nodes]
        
        # Step 2: Evaluate passage relevance
        relevance_analysis = self.evaluate_passage_relevance(user_query, passages)
        
        # Step 3: Evaluate time sensitivity
        time_sensitivity_analysis = self.evaluate_time_sensitivity(user_query)
        
        # Step 4: Make the decision
        decision = self.make_decision(
            user_query, 
            relevance_analysis, 
            time_sensitivity_analysis
        )
        
        # Step 5: Get the response based on the decision
        if decision.get("make_external_request", True):
            # Use Perplexity for external information
            perplexity_chain = self.perplexity_prompt | self.perplexity
            response = perplexity_chain.invoke({"query": user_query})
            source = "external"
        else:
            # Use internal knowledge base
            response = self.index.as_query_engine().query(user_query)
            source = "internal"
        
        # Log the decision process if enabled
        if self.enable_logging:
            self._log_decision(
                user_query, 
                passages, 
                relevance_analysis, 
                time_sensitivity_analysis, 
                decision, 
                source
            )
        
        # Return the final result with metadata
        return {
            "query": user_query,
            "response": response,
            "source": source,
            "metadata": {
                "relevance_analysis": relevance_analysis,
                "time_sensitivity_analysis": time_sensitivity_analysis,
                "decision": decision,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _log_decision(self, query, passages, relevance_analysis, 
                     time_sensitivity_analysis, decision, source):
        """Log the decision process for analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "passages_count": len(passages),
            "relevance_analysis": relevance_analysis,
            "time_sensitivity_analysis": time_sensitivity_analysis,
            "decision": decision,
            "source_used": source
        }
        
        # In a real implementation, you might write to a file or database
        print(f"Decision Log: {json.dumps(log_entry, indent=2)}")

# Example usage
if __name__ == "__main__":
    # Initialize the SmartRAG system
    rag = SmartRAG()
    
    # Example queries
    queries = [
        # "What are the three most common fusion methods in multimodal large models?"
        # "Give me some details about Trump's tariff policy.",
        "Please only search for some information about Danish` culture and history through this website: https://denmark.dk/people-and-culture",
    ]
    
    # Process each query
    for query in queries:
        print(f"\n\n=== Processing Query: {query} ===")
        result = rag.query(query)
        print(f"Source: {result['source']}")
        if result['source'] == "internal":
            print("Internal Knowledge Base Response:")
            print(result['response'])
            
        else:
            print("External Network Response:")
            print(result['response'].content)
            print("Citations:")
            print(result['response'].additional_kwargs.get("citations", []))
