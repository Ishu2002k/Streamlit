from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# global vectorstore for few-shot examples
EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
)

# FAISS index setup
dimension = len(embedding_model.embed_query("test"))  # vector dimension
index = faiss.IndexFlatL2(dimension)

# Build empty FAISS store
fewshot_store = FAISS(
    embedding_model.embed_query, 
    index, 
    InMemoryDocstore({}), 
    {}
)

# Global memory (can swap to VectorStoreRetrieverMemory later)
conv_memory = ConversationBufferMemory(
    return_messages=True,
    memory_key = "chat_history",
    max_token_limit = 2000
)

# Example schema for prompt
example_template = PromptTemplate(
    input_variables = ["query","sql"],
    template = "User Query: {Query}\nSQL Query: {sql}"
)

def add_fewshot_example(query: str,sql: str):
    """Store successful query -> SQL in vectorstore"""
    try:
        doc = Document(
            page_content = query,
            metadata = {"query": query, "sql": sql}  # Fixed: store both query and sql
        )
        fewshot_store.add_documents([doc])
        print(f"✅ Added few-shot example: {query[:50]}...")
    except Exception as e:
        print(f"❌ Error adding few-shot example: {e}")

def get_fewshot_examples(user_query:str,k: int = 2):
    """Retrieve top-k relevant examples for few-shot prompting"""
    try:
        if fewshot_store.index.ntotal == 0:
            return []
        
        docs = fewshot_store.similarity_search(user_query, k=k)
        examples = []
        
        for doc in docs:
            if "query" in doc.metadata and "sql" in doc.metadata:
                examples.append({
                    "query": doc.metadata["query"], 
                    "sql": doc.metadata["sql"]
                })
        
        return examples
    except Exception as e:
        print(f"❌ Error retrieving few-shot examples: {e}")
        return []

def build_fewshot_prompt(user_query: str):
    """Build few-shot prompt with relevant examples"""
    examples = get_fewshot_examples(user_query, k=3)  # Increased to 3 for better context
    
    if not examples:
        return "No similar examples found yet.\n"
    
    try:
        fewshot_template = FewShotPromptTemplate(
            example_prompt=example_template,
            examples=examples,
            prefix="Here are some similar successful query examples:\n",
            suffix="Now generate SQL for the following query:\n",
            input_variables=[],  # No input variables needed for this use case
        )
        
        formatted_prompt = fewshot_template.format()
        print(f"📚 Using {len(examples)} few-shot examples for context")
        return formatted_prompt
        
    except Exception as e:
        print(f"❌ Error building few-shot prompt: {e}")
        return "Few-shot examples unavailable.\n"

def save_turn(user_input: str, sql: str, success: bool = True):
    """Store user input and SQL output in conversation memory"""
    try:
        if success:
            output = f"✅ Generated SQL: {sql}"
        else:
            output = f"❌ Failed SQL attempt: {sql}"
            
        conv_memory.save_context(
            {"input": user_input}, 
            {"output": output}
        )
        print(f"💾 Saved conversation turn: {user_input[:30]}...")
    except Exception as e:
        print(f"❌ Error saving conversation turn: {e}")

def get_conversation_context(max_turns: int = 3):
    """Retrieve recent conversation history for context injection"""
    try:
        history = conv_memory.load_memory_variables({})
        chat_history = history.get("chat_history", [])
        
        if not chat_history:
            return "No previous conversation context.\n"
        
        # Get last few turns (limit to prevent context overflow)
        recent_history = chat_history[-max_turns*2:] if len(chat_history) > max_turns*2 else chat_history
        
        context_lines = []
        for i, message in enumerate(recent_history):
            if hasattr(message, 'content'):
                role = "Human" if i % 2 == 0 else "Assistant"
                context_lines.append(f"{role}: {message.content}")
        
        if context_lines:
            context = "Recent conversation context:\n" + "\n".join(context_lines) + "\n"
            print(f"🔄 Using conversation context from {len(context_lines)} messages")
            return context
        else:
            return "No previous conversation context.\n"
            
    except Exception as e:
        print(f"❌ Error retrieving conversation context: {e}")
        return "Conversation context unavailable.\n"

def clear_conversation_memory():
    """Clear conversation memory - useful for testing or new sessions"""
    try:
        conv_memory.clear()
        print("🧹 Conversation memory cleared")
    except Exception as e:
        print(f"❌ Error clearing conversation memory: {e}")

def get_fewshot_stats():
    """Get statistics about stored few-shot examples"""
    try:
        total_examples = fewshot_store.index.ntotal
        return {
            "total_examples": total_examples,
            "status": "ready" if total_examples > 0 else "empty"
        }
    except Exception as e:
        print(f"❌ Error getting few-shot stats: {e}")
        return {"total_examples": 0, "status": "error"}

# Initialize with some basic examples (optional)
def initialize_with_base_examples():
    """Initialize the few-shot store with some basic SQL examples"""
    base_examples = [
        ("Count all records", "SELECT COUNT(*) FROM table_name"),
        ("Get top 10 records", "SELECT * FROM table_name LIMIT 10"),
        ("Group by and count", "SELECT column_name, COUNT(*) FROM table_name GROUP BY column_name"),
        ("Simple join", "SELECT * FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id")
    ]
    
    for query, sql in base_examples:
        add_fewshot_example(query, sql)
    
    print(f"🚀 Initialized with {len(base_examples)} base examples")

# ---------------------------------------------------------------------------------------------------
# Add these functions to your fewshot_adapter.py for error_handler_node function 
# ---------------------------------------------------------------------------------------------------
def add_error_correction_example(original_query: str, broken_sql: str, error_msg: str, fixed_sql: str):
    """Store error correction patterns for future reference"""
    try:
        error_pattern = f"Query: {original_query}\nBroken SQL: {broken_sql}\nError: {error_msg}\nFixed SQL: {fixed_sql}"
        doc = Document(
            page_content=f"Error correction for: {original_query}",
            metadata={
                "query": original_query,
                "broken_sql": broken_sql,
                "error_msg": error_msg,
                "fixed_sql": fixed_sql,
                "type": "error_correction"
            }
        )
        fewshot_store.add_documents([doc])
        print(f"✅ Added error correction pattern: {error_msg[:30]}...")
    except Exception as e:
        print(f"❌ Error storing correction pattern: {e}")

def get_error_correction_examples(error_msg: str, k: int = 2):
    """Retrieve similar error correction patterns"""
    try:
        if fewshot_store.index.ntotal == 0:
            return []
        
        # Search for similar error messages
        docs = fewshot_store.similarity_search(f"Error correction: {error_msg}", k=k)
        corrections = []
        
        for doc in docs:
            if doc.metadata.get("type") == "error_correction":
                corrections.append({
                    "query": doc.metadata.get("query", ""),
                    "broken_sql": doc.metadata.get("broken_sql", ""),
                    "error_msg": doc.metadata.get("error_msg", ""),
                    "fixed_sql": doc.metadata.get("fixed_sql", "")
                })
        
        return corrections
    except Exception as e:
        print(f"❌ Error retrieving correction examples: {e}")
        return []

def build_error_correction_prompt(original_query: str, error_msg: str):
    """Build specialized prompt for error correction with similar error patterns"""
    error_examples = get_error_correction_examples(error_msg, k=2)
    
    if not error_examples:
        return "No similar error correction patterns found.\n"
    
    try:
        correction_lines = ["Here are similar error correction examples:\n"]
        
        for i, example in enumerate(error_examples, 1):
            correction_lines.append(f"Example {i}:")
            correction_lines.append(f"Query: {example['query']}")
            correction_lines.append(f"Error: {example['error_msg']}")
            correction_lines.append(f"Broken SQL: {example['broken_sql']}")
            correction_lines.append(f"Fixed SQL: {example['fixed_sql']}\n")
        
        print(f"🔧 Using {len(error_examples)} error correction examples")
        return "\n".join(correction_lines)
        
    except Exception as e:
        print(f"❌ Error building correction prompt: {e}")
        return "Error correction examples unavailable.\n"
    













