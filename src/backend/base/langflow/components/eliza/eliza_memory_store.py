"""Eliza Memory Store Component for LangFlow."""

from typing import Any, Dict, List, Optional

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs.inputs import (
    BoolInput,
    DictInput,
    FloatInput,
    HandleInput,
    MessageTextInput,
    MultilineInput,
    StrInput,
)
from langflow.io import Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class ElizaMemoryStoreComponent(LCToolComponent):
    """
    LangFlow component for storing memories in Eliza's memory system.
    
    This component provides:
    - Manual memory storage
    - Memory association creation
    - Memory management operations
    - Batch memory operations
    """
    
    display_name = "Eliza Memory Store"
    description = "Store and manage memories in Eliza's long-term memory system."
    name = "eliza_memory_store"
    icon = "💾"
    
    inputs = [
        MessageTextInput(
            name="content",
            display_name="Memory Content",
            info="The content to store as a memory.",
        ),
        StrInput(
            name="user_id",
            display_name="User ID",
            info="User ID to store the memory for.",
            value="anonymous",
        ),
        StrInput(
            name="session_id",
            display_name="Session ID",
            info="Session ID to associate with the memory.",
            value="default_session",
        ),
        StrInput(
            name="memory_type",
            display_name="Memory Type",
            info="Type of memory (e.g., 'preference', 'factual', 'contextual', 'general').",
            value="general",
        ),
        MultilineInput(
            name="tags",
            display_name="Tags",
            info="Comma-separated list of tags for the memory.",
            value="",
        ),
        DictInput(
            name="metadata",
            display_name="Metadata",
            info="Additional metadata to store with the memory.",
            is_list=False,
        ),
        FloatInput(
            name="relevance_score",
            display_name="Relevance Score",
            info="Relevance score for the memory (0.0 to 1.0).",
            value=1.0,
        ),
        BoolInput(
            name="generate_embedding",
            display_name="Generate Embedding",
            info="Whether to generate vector embedding for semantic search.",
            value=True,
        ),
        HandleInput(
            name="memory_instance",
            display_name="Memory Instance",
            input_types=["ElizaLongTermMemory"],
            info="Eliza memory instance to store in.",
        ),
    ]
    
    outputs = [
        Output(
            display_name="Stored Memory",
            name="stored_memory",
            method="store_memory",
        ),
        Output(
            display_name="Memory ID",
            name="memory_id",
            method="get_memory_id",
        ),
        Output(
            display_name="Storage Status",
            name="storage_status",
            method="get_storage_status",
        ),
        Output(
            display_name="Updated Memory Instance",
            name="updated_memory_instance",
            method="get_updated_memory_instance",
        ),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stored_memory = None
        self._memory_id = None
        self._storage_status = "Not attempted"
    
    def store_memory(self) -> Data:
        """Store the memory and return the stored memory data."""
        if not self.memory_instance:
            self._storage_status = "Error: No memory instance provided"
            return Data(data={})
        
        if not self.content:
            self._storage_status = "Error: No content provided"
            return Data(data={})
        
        try:
            # Parse tags
            tags = []
            if self.tags:
                tags = [t.strip() for t in self.tags.split(",") if t.strip()]
            
            # Store the memory
            if hasattr(self.memory_instance, 'add_memory'):
                # Use the add_memory method if available
                self._memory_id = self.memory_instance.add_memory(
                    content=self.content,
                    memory_type=self.memory_type,
                    metadata=self.metadata,
                    tags=tags
                )
                
                # Try to get the stored memory data
                if hasattr(self.memory_instance, 'memory_store'):
                    stored_memories = [
                        m for m in self.memory_instance.memory_store.memories 
                        if m.get('id') == self._memory_id
                    ]
                    if stored_memories:
                        self._stored_memory = Data(data=stored_memories[0])
                
            elif hasattr(self.memory_instance, 'store_memory'):
                # Use the store_memory method if available (for database-backed instances)
                stored_memory_obj = self.memory_instance.store_memory(
                    user_id=self.user_id,
                    content=self.content,
                    memory_type=self.memory_type,
                    session_id=self.session_id,
                    metadata=self.metadata,
                    tags=tags,
                    relevance_score=self.relevance_score
                )
                
                if stored_memory_obj:
                    if hasattr(stored_memory_obj, 'to_dict'):
                        self._stored_memory = Data(data=stored_memory_obj.to_dict())
                        self._memory_id = str(stored_memory_obj.id)
                    else:
                        self._stored_memory = Data(data={"id": str(stored_memory_obj)})
                        self._memory_id = str(stored_memory_obj)
            
            else:
                self._storage_status = "Error: Memory instance does not support memory storage"
                return Data(data={})
            
            if self._memory_id:
                self._storage_status = f"Successfully stored memory with ID: {self._memory_id}"
            else:
                self._storage_status = "Memory stored but ID not available"
            
            return self._stored_memory or Data(data={
                "content": self.content,
                "memory_type": self.memory_type,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "tags": tags,
                "metadata": self.metadata,
                "relevance_score": self.relevance_score
            })
            
        except Exception as e:
            self._storage_status = f"Error storing memory: {str(e)}"
            return Data(data={})
    
    def get_memory_id(self) -> Message:
        """Get the ID of the stored memory."""
        self.store_memory()
        return Message(text=self._memory_id or "No memory ID available")
    
    def get_storage_status(self) -> Message:
        """Get the storage status message."""
        self.store_memory()
        return Message(text=self._storage_status)
    
    def get_updated_memory_instance(self) -> Any:
        """Get the updated memory instance after storing."""
        self.store_memory()
        return self.memory_instance

