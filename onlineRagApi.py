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

# FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI(
    title="SmartRAG API",
    description="A RAG system that intelligently decides when to make external network requests based on query analysis.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

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
    
    async def evaluate_passage_relevance(self, query: str, passages: List[str]) -> Dict[str, Any]:
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
    
    async def evaluate_time_sensitivity(self, query: str) -> Dict[str, Any]:
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
    
    async def make_decision(self, query: str, relevance_analysis: Dict[str, Any], 
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
    
    async def query_streaming(self, user_query: str):
        """
        Main query method that orchestrates the entire process with streaming results.
        
        Args:
            user_query: The user's query
            
        Yields:
            Dict containing intermediate and final results
        """
        # Step 1: Retrieve passages from internal knowledge base
        yield {"status": "processing", "step": "retrieving", "message": "Retrieving information from knowledge base..."}
        
        nodes = self.retriever.retrieve(user_query)
        passages = [node.text for node in nodes]
        
        yield {"status": "processing", "step": "retrieved", "message": f"Retrieved {len(passages)} passages"}
        
        # Step 2: Evaluate passage relevance
        yield {"status": "processing", "step": "analyzing_relevance", "message": "Evaluating relevance of information..."}
        
        relevance_analysis = await self.evaluate_passage_relevance(user_query, passages)
        
        yield {
            "status": "processing", 
            "step": "relevance_analyzed", 
            "message": f"Relevance score: {relevance_analysis.get('relevance_score', 0):.2f}",
            "data": relevance_analysis
        }
        
        # Step 3: Evaluate time sensitivity
        yield {"status": "processing", "step": "analyzing_time_sensitivity", "message": "Evaluating time sensitivity..."}
        
        time_sensitivity_analysis = await self.evaluate_time_sensitivity(user_query)
        
        yield {
            "status": "processing", 
            "step": "time_sensitivity_analyzed", 
            "message": f"Time sensitivity score: {time_sensitivity_analysis.get('time_sensitive', 0):.2f}",
            "data": time_sensitivity_analysis
        }
        
        # Step 4: Make the decision
        yield {"status": "processing", "step": "deciding", "message": "Making decision on information source..."}
        
        decision = await self.make_decision(
            user_query, 
            relevance_analysis, 
            time_sensitivity_analysis
        )
        
        yield {
            "status": "processing", 
            "step": "decision_made", 
            "message": f"Decision: {'External request' if decision.get('make_external_request', True) else 'Internal knowledge base'}",
            "data": decision
        }
        
        # Step 5: Get the response based on the decision
        if decision.get("make_external_request", True):
            # Use Perplexity for external information
            yield {"status": "processing", "step": "external_request", "message": "Making external request for up-to-date information..."}
            
            perplexity_chain = self.perplexity_prompt | self.perplexity
            response = perplexity_chain.invoke({"query": user_query})
            source = "external"
            final_response = response.content
            citations = response.additional_kwargs.get("citations", [])
            
            yield {
                "status": "complete", 
                "source": source, 
                "response": final_response,
                "citations": citations
            }
        else:
            # Use internal knowledge base
            yield {"status": "processing", "step": "internal_response", "message": "Generating response from internal knowledge base..."}
            
            response = self.index.as_query_engine().query(user_query)
            source = "internal"
            final_response = str(response)
            
            yield {
                "status": "complete", 
                "source": source, 
                "response": final_response
            }
        
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


# Create a singleton instance of SmartRAG
rag = SmartRAG()

@app.get("/")
async def root():
    return {"message": "SmartRAG API is running", "docs": "/docs"}

@app.post("/query")
async def query(request: QueryRequest):
    """
    Non-streaming endpoint that returns the complete response.
    This is useful for simple queries or testing.
    """
    
    # ==========================Please Modify it follow the instruction========================================
    
    
    # Step 1: Retrieve passages from internal knowledge base
    nodes = rag.retriever.retrieve(request.query)
    passages = [node.text for node in nodes]
    
    # Step 2: Evaluate passage relevance
    relevance_analysis = await rag.evaluate_passage_relevance(request.query, passages)
    
    # Step 3: Evaluate time sensitivity
    time_sensitivity_analysis = await rag.evaluate_time_sensitivity(request.query)
    
    # Step 4: Make the decision
    decision = await rag.make_decision(
        request.query, 
        relevance_analysis, 
        time_sensitivity_analysis
    )
    
    # Step 5: Get the response based on the decision
    if decision.get("make_external_request", True):
        # Use Perplexity for external information
        perplexity_chain = rag.perplexity_prompt | rag.perplexity
        response = perplexity_chain.invoke({"query": request.query})
        source = "external"
        final_response = response.content
        citations = response.additional_kwargs.get("citations", [])
        
        return {
            "query": request.query,
            "source": source,
            "response": final_response,
            "citations": citations,
            "metadata": {
                "relevance_analysis": relevance_analysis,
                "time_sensitivity_analysis": time_sensitivity_analysis,
                "decision": decision,
                "timestamp": datetime.now().isoformat()
            }
        }
    else:
        # Use internal knowledge base
        response = rag.index.as_query_engine().query(request.query)
        source = "internal"
        
        return {
            "query": request.query,
            "source": source,
            "response": str(response),
            "metadata": {
                "relevance_analysis": relevance_analysis,
                "time_sensitivity_analysis": time_sensitivity_analysis,
                "decision": decision,
                "timestamp": datetime.now().isoformat()
            }
        }

@app.get("/query/stream")
async def query_stream(query: str):
    """
    Streaming endpoint that returns results as they become available.
    This is useful for showing the user the reasoning process in real-time.
    """
    async def stream_generator():
        async for item in rag.query_streaming(query):
            # Convert to JSON and yield with newline for proper streaming
            yield json.dumps(item) + "\n"
            # Small delay to ensure proper streaming in the browser
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    uvicorn.run("onlineRagApi:app", host="0.0.0.0", port=8000, reload=True)