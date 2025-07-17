"""Eliza Memory Component for LangFlow."""

from typing import Any, Dict, List, Optional

from langflow.base.memory.model import LCChatMemoryComponent
from langflow.inputs.inputs import (
    BoolInput,
    DictInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    StrInput,
)
from langflow.io import Output
from langflow.schema.data import Data
from langflow.schema.message import Message


class ElizaMemoryComponent(LCChatMemoryComponent):
    """
    LangFlow component for Eliza's long-term memory system.
    
    This component provides:
    - Persistent conversation storage
    - Memory extraction and retrieval
    - Semantic search capabilities
    - Integration with LangChain memory classes
    """
    
    display_name = "Eliza Memory"
    description = "Advanced long-term memory system for Eliza with semantic search and persistent storage."
    name = "eliza_memory"
    icon = "🧠"
    
    inputs = [
        StrInput(
            name="session_id",
            display_name="Session ID",
            info="Unique identifier for the conversation session.",
            value="default_session",
        ),
        StrInput(
            name="user_id",
            display_name="User ID",
            info="Unique identifier for the user.",
            value="anonymous",
        ),
        MessageTextInput(
            name="message",
            display_name="Message",
            info="The message to process and store in memory.",
        ),
        BoolInput(
            name="extract_memories",
            display_name="Extract Memories",
            info="Whether to automatically extract and store important information as memories.",
            value=True,
        ),
        BoolInput(
            name="include_context",
            display_name="Include Memory Context",
            info="Whether to include relevant memories in the response context.",
            value=True,
        ),
        IntInput(
            name="max_memories",
            display_name="Max Memories to Retrieve",
            info="Maximum number of relevant memories to include in context.",
            value=5,
        ),
        DictInput(
            name="metadata",
            display_name="Metadata",
            info="Additional metadata to store with the message.",
            is_list=False,
        ),
        HandleInput(
            name="existing_memory",
            display_name="Existing Memory Instance",
            input_types=["ElizaLongTermMemory"],
            info="Existing Eliza memory instance to use.",
        ),
    ]
    
    outputs = [
        Output(
            display_name="Memory Context",
            name="memory_context",
            method="get_memory_context",
        ),
        Output(
            display_name="Memory Instance",
            name="memory_instance",
            method="get_memory_instance",
        ),
        Output(
            display_name="Stored Message",
            name="stored_message",
            method="get_stored_message",
        ),
        Output(
            display_name="Relevant Memories",
            name="relevant_memories",
            method="get_relevant_memories",
        ),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._memory_instance = None
        self._memory_context = ""
        self._stored_message = None
        self._relevant_memories = []
    
    def build_memory(self) -> Any:
        """Build and return the Eliza memory instance."""
        try:
            # Import here to avoid circular imports
            from langchain.memory.eliza_memory import create_eliza_memory
            
            if self.existing_memory:
                self._memory_instance = self.existing_memory
            else:
                self._memory_instance = create_eliza_memory(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    memory_extraction=self.extract_memories
                )
            
            # Process the message if provided
            if self.message:
                self._process_message()
            
            return self._memory_instance
            
        except Exception as e:
            self.status = f"Error building memory: {str(e)}"
            return None
    
    def _process_message(self):
        """Process the input message and update memory."""
        if not self._memory_instance or not self.message:
            return
        
        try:
            # Store the message in conversation history
            inputs = {"input": self.message}
            outputs = {"output": ""}  # Will be filled by the chain
            
            # Add metadata if provided
            if self.metadata:
                inputs.update(self.metadata)
            
            # Save context to memory
            self._memory_instance.save_context(inputs, outputs)
            
            # Get relevant memories if context is enabled
            if self.include_context:
                self._relevant_memories = self._memory_instance.search_memories(
                    self.message, 
                    limit=self.max_memories
                )
                
                # Load memory variables for context
                memory_vars = self._memory_instance.load_memory_variables(inputs)
                self._memory_context = memory_vars.get(self._memory_instance.memory_key, "")
            
            # Create stored message data
            self._stored_message = Data(
                data={
                    "content": self.message,
                    "session_id": self.session_id,
                    "user_id": self.user_id,
                    "timestamp": self._memory_instance._get_timestamp(),
                    "metadata": self.metadata or {}
                }
            )
            
        except Exception as e:
            self.status = f"Error processing message: {str(e)}"
    
    def get_memory_context(self) -> Message:
        """Get the memory context as a message."""
        self.build_memory()
        return Message(text=self._memory_context or "No memory context available.")
    
    def get_memory_instance(self) -> Any:
        """Get the memory instance."""
        return self.build_memory()
    
    def get_stored_message(self) -> Data:
        """Get the stored message data."""
        self.build_memory()
        return self._stored_message or Data(data={})
    
    def get_relevant_memories(self) -> List[Data]:
        """Get relevant memories as a list of Data objects."""
        self.build_memory()
        
        memory_data = []
        for memory in self._relevant_memories:
            memory_data.append(Data(data=memory))
        
        return memory_data

