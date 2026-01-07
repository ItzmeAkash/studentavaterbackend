import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

load_dotenv()

router = APIRouter()


# Request model for chat
class ChatRequest(BaseModel):
    message: str  # Simple message string


class ChatResponse(BaseModel):
    response: str  # Simple response string


def _get_chat_model():
    """Initialize and return ChatGroq model"""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured in environment variables",
        )

    try:
        # Initialize ChatGroq with default model
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", temperature=0.7, groq_api_key=api_key
        )
        return llm
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize ChatGroq: {str(e)}"
        )


def _create_prompt_template(system_prompt: Optional[str] = None):
    """Create ChatPromptTemplate with system message"""
    default_system = (
        "You are a helpful AI assistant. Be concise and helpful in your responses."
    )
    system_message = system_prompt if system_prompt else default_system

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    return prompt


def _parse_messages(
    messages: List[dict],
) -> Tuple[Optional[str], List[BaseMessage], str]:
    """
    Parse messages list to extract system prompt, conversation history, and current user message

    Returns:
        tuple: (system_prompt, chat_history, current_user_message)
    """
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    system_prompt = None
    chat_history: List[BaseMessage] = []
    current_user_message = ""

    # Check if first message is a system message
    if messages[0].get("role", "").lower() == "system":
        system_prompt = messages[0].get("content", "")
        messages = messages[1:]  # Remove system message from history

    if not messages:
        raise HTTPException(
            status_code=400, detail="At least one user message is required"
        )

    # Get the last message as current user message
    last_message = messages[-1]
    last_role = last_message.get("role", "").lower()

    if last_role not in ["user", "human"]:
        raise HTTPException(status_code=400, detail="Last message must be from user")

    current_user_message = last_message.get("content", "")

    # Convert remaining messages (except last) to conversation history
    for msg in messages[:-1]:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")

        if role in ["user", "human"]:
            chat_history.append(HumanMessage(content=content))
        elif role in ["assistant", "ai"]:
            chat_history.append(AIMessage(content=content))

    return system_prompt, chat_history, current_user_message


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Simple chat API endpoint using LangChain Groq with ChatPromptTemplate

    Args:
        request: ChatRequest containing a simple message string: {"message": "hello"}

    Returns:
        ChatResponse with simple response string: {"response": "..."}
    """
    try:
        # Get the chat model
        llm = _get_chat_model()

        # Create prompt template (no conversation history for simple chat)
        prompt_template = _create_prompt_template()

        # Create the chain
        chain = prompt_template | llm

        # Invoke the chain with empty history
        response = await chain.ainvoke({
            "input": request.message,
            "chat_history": [],
        })

        # Extract the response content
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        return ChatResponse(response=response_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/chat/health")
def chat_health():
    """Health check for chat endpoint"""
    return {
        "status": "healthy",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }
