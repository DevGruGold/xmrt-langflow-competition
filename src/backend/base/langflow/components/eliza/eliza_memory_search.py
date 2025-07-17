"""Eliza Memory Search Component for LangFlow."""

from typing import Any, Dict, List, Optional

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs.inputs import (
    BoolInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    StrInput,
)
from langflow.io import Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class ElizaMemorySearchComponent(LCToolComponent):
    """
    LangFlow component for searching Eliza's memory system.
    
    This component provides:
    - Semantic memory search
    - Keyword-based search
    - Memory filtering by type and tags
    - Memory analytics and insights
    """
    
    display_name = "Eliza Memory Search"
    description = "Search and retrieve memories from Eliza's long-term memory system."
    name = "eliza_memory_search"
    icon = "🔍"
    
    inputs = [
        MessageTextInput(
            name="search_query",
            display_name="Search Query",
            info="The query to search for in memories.",
        ),
        StrInput(
            name="user_id",
            display_name="User ID",
            info="User ID to search memories for.",
            value="anonymous",
        ),
        MultilineInput(
            name="memory_types",
            display_name="Memory Types",
            info="Comma-separated list of memory types to filter by (e.g., 'preference,factual,contextual').",
            value="",
        ),
        MultilineInput(
            name="tags",
            display_name="Tags",
            info="Comma-separated list of tags to filter by.",
            value="",
        ),
        IntInput(
            name="limit",
            display_name="Result Limit",
            info="Maximum number of memories to return.",
            value=10,
        ),
        BoolInput(
            name="semantic_search",
            display_name="Use Semantic Search",
            info="Whether to use semantic similarity search instead of keyword search.",
            value=True,
        ),
        BoolInput(
            name="include_analytics",
            display_name="Include Analytics",
            info="Whether to include memory analytics in the results.",
            value=False,
        ),
        HandleInput(
            name="memory_instance",
            display_name="Memory Instance",
            input_types=["ElizaLongTermMemory"],
            info="Eliza memory instance to search.",
        ),
    ]
    
    outputs = [
        Output(
            display_name="Search Results",
            name="search_results",
            method="search_memories",
        ),
        Output(
            display_name="Formatted Results",
            name="formatted_results",
            method="get_formatted_results",
        ),
        Output(
            display_name="Memory Analytics",
            name="memory_analytics",
            method="get_memory_analytics",
        ),
        Output(
            display_name="Result Count",
            name="result_count",
            method="get_result_count",
        ),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._search_results = []
        self._analytics = {}
    
    def search_memories(self) -> List[Data]:
        """Search memories and return results."""
        if not self.memory_instance:
            self.status = "No memory instance provided"
            return []
        
        if not self.search_query:
            self.status = "No search query provided"
            return []
        
        try:
            # Parse memory types and tags
            memory_types = []
            if self.memory_types:
                memory_types = [t.strip() for t in self.memory_types.split(",") if t.strip()]
            
            tags = []
            if self.tags:
                tags = [t.strip() for t in self.tags.split(",") if t.strip()]
            
            # Perform search based on type
            if self.semantic_search and hasattr(self.memory_instance, 'search_memories_semantic'):
                # Semantic search with similarity scores
                results = self.memory_instance.search_memories_semantic(
                    self.search_query,
                    limit=self.limit,
                    memory_types=memory_types
                )
                
                # Convert to Data objects with similarity scores
                self._search_results = []
                for memory, similarity in results:
                    memory_data = memory.copy() if isinstance(memory, dict) else memory
                    if hasattr(memory_data, 'to_dict'):
                        memory_data = memory_data.to_dict()
                    memory_data['similarity_score'] = similarity
                    self._search_results.append(Data(data=memory_data))
                    
            else:
                # Regular keyword search
                results = self.memory_instance.search_memories(
                    self.search_query,
                    memory_types=memory_types,
                    limit=self.limit
                )
                
                # Convert to Data objects
                self._search_results = []
                for memory in results:
                    memory_data = memory.copy() if isinstance(memory, dict) else memory
                    if hasattr(memory_data, 'to_dict'):
                        memory_data = memory_data.to_dict()
                    self._search_results.append(Data(data=memory_data))
            
            # Get analytics if requested
            if self.include_analytics and hasattr(self.memory_instance, 'get_memory_analytics'):
                self._analytics = self.memory_instance.get_memory_analytics()
            
            self.status = f"Found {len(self._search_results)} memories"
            return self._search_results
            
        except Exception as e:
            self.status = f"Error searching memories: {str(e)}"
            return []
    
    def get_formatted_results(self) -> Message:
        """Get formatted search results as a message."""
        self.search_memories()
        
        if not self._search_results:
            return Message(text="No memories found for the given query.")
        
        # Format results as text
        formatted_text = f"Found {len(self._search_results)} memories:\n\n"
        
        for i, result in enumerate(self._search_results, 1):
            data = result.data
            content = data.get('content', 'No content')
            memory_type = data.get('memory_type', 'unknown')
            similarity = data.get('similarity_score')
            
            formatted_text += f"{i}. [{memory_type.upper()}] {content}"
            
            if similarity is not None:
                formatted_text += f" (similarity: {similarity:.2f})"
            
            formatted_text += "\n\n"
        
        return Message(text=formatted_text)
    
    def get_memory_analytics(self) -> Data:
        """Get memory analytics data."""
        self.search_memories()
        return Data(data=self._analytics)
    
    def get_result_count(self) -> Message:
        """Get the number of search results."""
        self.search_memories()
        return Message(text=str(len(self._search_results)))

