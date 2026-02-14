import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import app
import uuid

st.set_page_config(page_title="SkyRalph - CRM Agent", page_icon="💬", layout="wide")

st.title("💬 SkyRalph - CRM Assistant")
st.markdown("""
**I am your AI analytics partner.** I can help you understand win rates, forecast revenue, 
and detect anomalies in your pipeline using SkyGeni's advanced metrics engine.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I'm SkyRalph, your Senior Sales Strategy Partner. I'm here to provide high-leverage data intelligence to help our Sales Leadership and RevOps teams.\n\nWhat specific sales data or insights can I help you with today? For example, I can provide:\n\n*   **Key sales metrics summaries** (Win Rate, PQS, WRE, Segment Momentum)\n*   **Revenue forecasts**\n*   **Win rate trend explanations**\n*   **Ad-hoc analysis** on any specific sales data you need\n\nJust let me know what you're looking for!")
    ]

# Sidebar utilities
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            AIMessage(content="Hello! I'm SkyRalph, your Senior Sales Strategy Partner. I'm here to provide high-leverage data intelligence to help our Sales Leadership and RevOps teams.\n\nWhat specific sales data or insights can I help you with today? For example, I can provide:\n\n*   **Key sales metrics summaries** (Win Rate, PQS, WRE, Segment Momentum)\n*   **Revenue forecasts**\n*   **Win rate trend explanations**\n*   **Ad-hoc analysis** on any specific sales data you need\n\nJust let me know what you're looking for!")
        ]
        st.rerun()

# Initialize thread ID for LangGraph checkpointing (if we add memory later)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

def extract_message_content(message):
    """
    Robustly extract text from LangChain message content.
    Handles both simple strings and complex content block lists.
    """
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        # Extract text from content blocks
        text_parts = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    return str(message.content)

# Display chat messages
# We only want to show:
# 1. Human messages
# 2. The FINAL AI message for each turn (skipping intermediate thoughts/tool calls)
display_messages = []
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        display_messages.append(msg)
    elif isinstance(msg, AIMessage):
        # If this is the last message OR the next message is a HumanMessage, 
        # it's the final output of a turn.
        is_last = (i == len(st.session_state.messages) - 1)
        next_is_human = (i < len(st.session_state.messages) - 1 and isinstance(st.session_state.messages[i+1], HumanMessage))
        
        # Also skip messages that are purely tool calls without content
        content = extract_message_content(msg)
        if (is_last or next_is_human) and content.strip():
            display_messages.append(msg)

for message in display_messages:
    content = extract_message_content(message)
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask me about your sales pipeline..."):
    # Append user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate the response RIGHT HERE, before any rerun
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.status("SkyRalph is analyzing sales data...", expanded=True) as status:
        try:
            final_state = app.invoke({"messages": st.session_state.messages}, config=config)
            
            # Extract new messages for logging to st.status
            all_messages = final_state["messages"]
            new_messages = all_messages[len(st.session_state.messages):]
            
            for msg in new_messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        st.write(f"🛠️ Using tool: **{tc['name']}**")
                elif hasattr(msg, "tool_call_id"):
                    st.write("📊 Data retrieved successfully.")
            
            st.session_state.messages = all_messages
            status.update(label="Analysis complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Oops, I ran into an error.", state="error")
            st.error(f"Error: {e}")
            st.session_state.messages.append(AIMessage(content=f"I'm sorry, I encountered an error while processing your request: {e}"))
    
    # Rerun to normalize the UI using the clean display loop at the top
    st.rerun()
