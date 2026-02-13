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
    st.session_state.messages = []

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
for message in st.session_state.messages:
    content = extract_message_content(message)
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask me about your sales pipeline..."):
    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing pipeline data..."):
            # Prepare input format for LangGraph
            # LangGraph expects a dictionary with 'messages' key
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Streaming output to show tool calls
            response_placeholder = st.empty()
            full_response = ""
            
            # We use invoke for simplicity, but could use stream for better UX
            # To show tool usage, we can inspect the intermediate steps if we use stream
            # For now, let's just run invoke and show final answer
            
            # Run the graph
            final_state = app.invoke({"messages": st.session_state.messages}, config=config)
            
            # Extract new messages (LangGraph returns full history or delta depending on memory)
            # here we assume it returns the state which includes all messages
            all_messages = final_state["messages"]
            new_messages = all_messages[len(st.session_state.messages):]
            
            # Display intermediate tool calls if any
            for msg in new_messages:
                # 1. Display agent thought/text first if present
                content = extract_message_content(msg)
                if isinstance(msg, AIMessage) and content:
                    st.markdown(content)
                
                # 2. Display Tool Calls
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        with st.status(f"🛠️ Using tool: {tool_call['name']}", expanded=False) as status:
                            st.json(tool_call['args'])
                            status.update(state="complete")
                            
                # 3. Display Tool Outputs
                elif hasattr(msg, "tool_call_id"): # Check for ToolMessage
                     with st.expander(f"📊 Tool Output"):
                        try:
                            import json
                            content_json = json.loads(msg.content)
                            st.json(content_json)
                        except:
                            st.code(msg.content)

            
            # Update session state with ALL messages from the graph state
            # This ensures we keep the tool messages and tool outputs in history for context
            st.session_state.messages = all_messages
