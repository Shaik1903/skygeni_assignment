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
for message in st.session_state.messages:
    content = extract_message_content(message)
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            if "CHART_DATA:" in content:
                parts = content.split("CHART_DATA:", 1)
                if parts[0].strip():
                    st.markdown(parts[0].strip())
                try:
                    import json
                    import plotly.graph_objects as go
                    chart_json = json.loads(parts[1].strip())
                    fig = go.Figure(chart_json)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.code(parts[1].strip())
            else:
                st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask me about your sales pipeline..."):
    # Append user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    # Rerun to show user message immediately through the main loop
    st.rerun()

# If the last message is from the user, generate a response
if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
    with st.chat_message("assistant"):
        with st.spinner("Analyzing pipeline data..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Run the graph
            # Note: LangGraph ReAct agent returns the full history including tool calls
            final_state = app.invoke({"messages": st.session_state.messages}, config=config)
            
            # Extract new messages
            all_messages = final_state["messages"]
            new_messages = all_messages[len(st.session_state.messages):]
            
            # Display ONLY the new messages logic (intermediate tool calls)
            for msg in new_messages:
                # 1. Display agent thought/text first if present
                content = extract_message_content(msg)
                if isinstance(msg, AIMessage) and content:
                    if "CHART_DATA:" in content:
                        # Split text and chart data
                        parts = content.split("CHART_DATA:", 1)
                        if parts[0].strip():
                            st.markdown(parts[0].strip())
                        
                        try:
                            import json
                            import plotly.graph_objects as go
                            chart_json = json.loads(parts[1].strip())
                            fig = go.Figure(chart_json)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render chart: {e}")
                            st.code(parts[1].strip())
                    else:
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
                            # Handle potential large text outputs or non-JSON
                            st.text(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

            # Update session state
            st.session_state.messages = all_messages
            # Rerun to normalize the UI
            st.rerun()
