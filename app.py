import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from config import get_api_key, set_api_key

def csv_tool(filename: str):
    return create_csv_agent( 
        ChatOpenAI(temperature=0, model="gpt-4o-2024-05-13", api_key=get_api_key()),
        filename,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        pandas_kwargs={"low_memory": False},
    )

def ask_agent(agent, query: str, context: str) -> str:
    prompt = f"""
    Previous context: {context}

    Analyze the following query and provide a response in one of these formats:
    1. For tables: {{"table": {{"columns": ["col1", "col2", ...], "data": [[val1, val2, ...], ...]}}}}
    2. For bar charts: {{"bar": {{"x": ["A", "B", ...], "y": [25, 24, ...]}}}}
    3. For line charts: {{"line": {{"x": ["A", "B", ...], "y": [25, 24, ...]}}}}
    4. For scatter plots: {{"scatter": {{"x": [1, 2, ...], "y": [3, 4, ...]}}}}
    5. For pie charts: {{"pie": {{"labels": ["A", "B", ...], "values": [25, 24, ...]}}}}
    6. For histograms: {{"histogram": {{"x": [1, 2, 3, ...]}}}}
    7. For box plots: {{"box": {{"x": ["A", "B", ...], "y": [[1, 2, ...], [3, 4, ...], ...]}}}}
    8. For violin plots: {{"violin": {{"x": ["A", "B", ...], "y": [[1, 2, ...], [3, 4, ...], ...]}}}}
    9. For area charts: {{"area": {{"x": ["A", "B", ...], "y": [25, 24, ...]}}}}
    10. For heatmaps: {{"heatmap": {{"x": ["A", "B", ...], "y": ["X", "Y", ...], "z": [[1, 2, ...], [3, 4, ...], ...]}}}}
    11. For contour plots: {{"contour": {{"x": [1, 2, ...], "y": [1, 2, ...], "z": [[1, 2, ...], [3, 4, ...], ...]}}}}
    12. For 3D scatter plots: {{"scatter3d": {{"x": [1, 2, ...], "y": [3, 4, ...], "z": [5, 6, ...]}}}}
    13. For bubble charts: {{"bubble": {{"x": [1, 2, ...], "y": [3, 4, ...], "size": [10, 20, ...]}}}}
    14. For funnel charts: {{"funnel": {{"x": [25, 24, ...], "y": ["A", "B", ...]}}}}
    15. For treemaps: {{"treemap": {{"labels": ["A", "B", ...], "parents": ["", "A", ...], "values": [25, 24, ...]}}}}
    16. For sunburst charts: {{"sunburst": {{"labels": ["A", "B", ...], "parents": ["", "A", ...], "values": [25, 24, ...]}}}}
    17. For radar charts: {{"radar": {{"theta": ["A", "B", ...], "r": [25, 24, ...]}}}}
    18. For candlestick charts: {{"candlestick": {{"x": ["2023-01-01", ...], "open": [10, ...], "high": [15, ...], "low": [5, ...], "close": [12, ...]}}}}
    19. For distribution plots: {{"distribution": {{"x": [1, 2, 3, ...]}}}}
    20. For plain answers: {{"answer": "Your response here"}}

    Only use one of the above formats. If the query doesn't explicitly ask for a visualization or table, use the "answer" format.

    Query: {query}
    """
    response = agent.run(prompt)
    return str(response)

def decode_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"answer": response}

def explain_graph(agent, graph_type, data):
    prompt = f"""
    Explain the following {graph_type} graph data in 4 sentences or less:
    {json.dumps(data)}
    Focus on key insights, patterns, or notable features of the data.
    """
    explanation = agent.run(prompt)
    return explanation

def write_answer(agent, response_dict: dict):
    if "answer" in response_dict:
        st.write(response_dict["answer"])
        return

    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
        st.write(explain_graph(agent, "table", data))

    if "bar" in response_dict:
        data = response_dict["bar"]
        fig = px.bar(x=data["x"], y=data["y"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "bar", data))

    if "line" in response_dict:
        data = response_dict["line"]
        fig = px.line(x=data["x"], y=data["y"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "line", data))

    if "scatter" in response_dict:
        data = response_dict["scatter"]
        fig = px.scatter(x=data["x"], y=data["y"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "scatter", data))

    if "pie" in response_dict:
        data = response_dict["pie"]
        fig = px.pie(values=data["values"], names=data["labels"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "pie", data))

    if "histogram" in response_dict:
        data = response_dict["histogram"]
        fig = px.histogram(x=data["x"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "histogram", data))

    if "box" in response_dict:
        data = response_dict["box"]
        try:
            df = pd.DataFrame({col: values for col, values in zip(data["x"], data["y"])})
            fig = px.box(df)
            st.plotly_chart(fig)
            st.write(explain_graph(agent, "box", data))
        except ValueError as e:
            st.error(f"Unable to create box plot: {str(e)}. Please ensure the data is in the correct format.")

    if "violin" in response_dict:
        data = response_dict["violin"]
        try:
            df = pd.DataFrame({col: values for col, values in zip(data["x"], data["y"])})
            fig = px.violin(df)
            st.plotly_chart(fig)
            st.write(explain_graph(agent, "violin", data))
        except ValueError as e:
            st.error(f"Unable to create violin plot: {str(e)}. Please ensure the data is in the correct format.")

    if "area" in response_dict:
        data = response_dict["area"]
        fig = px.area(x=data["x"], y=data["y"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "area", data))

    if "heatmap" in response_dict:
        data = response_dict["heatmap"]
        fig = px.imshow(data["z"], x=data["x"], y=data["y"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "heatmap", data))

    if "contour" in response_dict:
        data = response_dict["contour"]
        fig = px.contour(x=data["x"], y=data["y"], z=data["z"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "contour", data))

    if "scatter3d" in response_dict:
        data = response_dict["scatter3d"]
        fig = px.scatter_3d(x=data["x"], y=data["y"], z=data["z"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "3D scatter", data))

    if "bubble" in response_dict:
        data = response_dict["bubble"]
        fig = px.scatter(x=data["x"], y=data["y"], size=data["size"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "bubble", data))

    if "funnel" in response_dict:
        data = response_dict["funnel"]
        fig = px.funnel(x=data["x"], y=data["y"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "funnel", data))

    if "treemap" in response_dict:
        data = response_dict["treemap"]
        fig = px.treemap(names=data["labels"], parents=data["parents"], values=data["values"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "treemap", data))

    if "sunburst" in response_dict:
        data = response_dict["sunburst"]
        fig = px.sunburst(names=data["labels"], parents=data["parents"], values=data["values"])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "sunburst", data))

    if "radar" in response_dict:
        data = response_dict["radar"]
        fig = px.line_polar(r=data["r"], theta=data["theta"], line_close=True)
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "radar", data))

    if "candlestick" in response_dict:
        data = response_dict["candlestick"]
        fig = go.Figure(data=[go.Candlestick(x=data["x"], open=data["open"], high=data["high"], low=data["low"], close=data["close"])])
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "candlestick", data))

    if "distribution" in response_dict:
        data = response_dict["distribution"]
        fig = px.histogram(x=data["x"], marginal="box")
        st.plotly_chart(fig)
        st.write(explain_graph(agent, "distribution", data))

def main():
    st.set_page_config(page_title="👨‍💻 Chat with Your CSV", layout="wide")

    # Center-align the title
    st.markdown("<h1 style='text-align: center;'>👨‍💻 Chat with Your CSV</h1>", unsafe_allow_html=True)

    # Create three columns, with the middle one being wider
    left_spacer, center_content, right_spacer = st.columns([1, 2, 1])

    # Use the center column for the API key input
    with center_content:
        # API key input code here
        api_key = get_api_key()
        if not api_key:
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if api_key:
                set_api_key(api_key)
                st.success("API key set successfully!")
                st.rerun()
            else:
                st.warning("Please enter your OpenAI API key to continue.")
                return
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory()

    # Create two columns
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Upload CSV")
        data = st.file_uploader("Upload a CSV file", type="csv")
        if data is not None and st.session_state.agent is None:
            try:
                st.session_state.agent = csv_tool(data)
                st.success("CSV file loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred while loading the CSV: {str(e)}")

    with col2:
        st.subheader("Chat")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        # Create a container for the input field at the bottom
        input_container = st.container()

        # Handle user input
        with input_container:
            # Use a callback to update session state
            def submit():
                st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
                st.session_state.user_input = ""  # Reset the input after submitting

            query = st.text_input("Ask a question about your CSV", key="user_input", on_change=submit)

            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                if st.session_state.agent is not None:
                    with st.spinner("Thinking..."):
                        # Get the conversation history
                        context = st.session_state.conversation_memory.buffer
                        
                        response = ask_agent(st.session_state.agent, st.session_state.messages[-1]["content"], context)
                        decoded_response = decode_response(response)
                        
                        # Add assistant's response to chat history
                        if "answer" in decoded_response:
                            st.session_state.messages.append({"role": "assistant", "content": decoded_response["answer"]})
                            # Update conversation memory
                            st.session_state.conversation_memory.save_context(
                                {"input": st.session_state.messages[-2]["content"]},
                                {"output": decoded_response["answer"]}
                            )
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": "Generated a visual response."})
                            # Update conversation memory
                            st.session_state.conversation_memory.save_context(
                                {"input": st.session_state.messages[-2]["content"]},
                                {"output": "Generated a visual response."}
                            )
                        
                        # Store the full response
                        st.session_state.responses.append(decoded_response)
                else:
                    st.error("Please upload a CSV file before asking questions.")

        # Display chat messages
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        write_answer(st.session_state.agent, st.session_state.responses[i // 2])
                    else:
                        st.write(message["content"])

if __name__ == "__main__":
    main()