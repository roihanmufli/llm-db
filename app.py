import streamlit as st
import os
from src.data_prep import Prep
from src.agent import SQLAgent
from src.plotly_agent import VizAgent
from src.layout import *
import tempfile
from pathlib import Path
import shutil
from sqlalchemy import create_engine
import plotly.express as px
import pandas as pd
from google.api_core.exceptions import ServiceUnavailable
import plotly.graph_objs as go
# import speech_recognition as sr
from streamlit_feedback import streamlit_feedback
import mysql.connector
from mysql.connector import Error
import uuid
from datetime import datetime
import base64


# extracting text from document

# @st.cache_resource(show_spinner=False)
def get_example_query():
    ingestion = Prep()
    vector_store = ingestion.get_query_examples()
    return vector_store

def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening... Please speak into the microphone.")
        audio = recognizer.listen(source)
    
    try:
        st.info("Recognizing speech...")
        text = recognizer.recognize_google(audio, language='id-ID')
        return text
    except sr.RequestError:
        return "API was unreachable or unresponsive."
    except sr.UnknownValueError:
        return "Unable to recognize speech."

def check_viz(query,selected_option,vector):
    if selected_option == "gemini-1.5-pro-latest":
        model = "models/gemini-1.5-pro-latest"
    else:
        model = "models/gemini-1.5-flash"
    viz_agent = VizAgent(model,vector)
    is_viz,kind_chart,title = viz_agent.check_viz_or_not(query)
    return is_viz,kind_chart,title 



def get_conversationchain(query,selected_option,vector):
    if selected_option == "gemini-1.5-pro-latest":
        model = "models/gemini-1.5-pro-latest"
    else:
        model = "models/gemini-1.5-flash"

    ## check visualization or not

    qna = SQLAgent(model,vector)
    results,source_query = qna.generate_response(
        query=query
    )
    return results,source_query

def create_connection():
    host_name = "mysql-14927681-techconnect.h.aivencloud.com"
    user_name = "avnadmin"
    user_password = os.getenv("DB_PASSWORD")
    db_name = "defaultdb"
    port = "13145"
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name,
            port=port
        )
        if connection.is_connected():
            print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

# Replace these with your actual database credentials


def store_feedback(question, response, response_type, user_feedback, timestamp, session_id):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT INTO feedback (question, response, response_type, user_feedback, timestamp, session_id) VALUES (%s, %s, %s, %s, %s, %s)''', 
            (question, response, response_type, user_feedback, timestamp, session_id))
        conn.commit()
        conn.close()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")
        conn.rollback()

# Feedback mechanism using streamlit-feedback
def handle_feedback(user_response,result):
    # st.write(f"Session ID: {st.session_state.session_id}")
    # # st.write("Result:", response)
    # st.write(f"User feedback: {user_response}")
    st.toast("✔️ Feedback received!")

    response_type = 'good' if user_response['score']=='👍' else 'bad'
    feedback = user_response['text']

    # Get the current timestamp
    timestamp = datetime.now()

    # Format the timestamp as a string in 'yyyymmddhhmmss' format
    timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')

    # Store feedback in the database
    store_feedback(st.session_state.messages[-2]["content"], str(result), response_type, feedback, timestamp_str, st.session_state.session_id)
          

    # Reset session ID after feedback is submitted
    st.session_state["session_id"] = str(uuid.uuid4())

# def clear_vector_db():
#     st.session_state.messages = [{"role": "assistant", "content": "upload some documents and ask me a question"}]
#     abs_path = os.path.dirname(os.path.abspath(__file__))
#     CurrentFolder = str(Path(abs_path).resolve())
#     path = os.path.join(CurrentFolder, "database")
#     shutil.rmtree(path)

# generating response from user queries and displaying them accordingly
# def handle_question(question):
#     response=st.session_state.conversation({'question': question})
#     st.session_state.chat_history=response["chat_history"]
#     for i,msg in enumerate(st.session_state.chat_history):
#         if i%2==0:
#             st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "ask me a question"}]
    
def main():
    st.set_page_config(page_title="Chat with multiple DOCUMENTs",page_icon="🤖")

    image_file = "NANO.png"
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read())
    st.markdown(
        f"""
        <style>
        [data-testid="stBottom"] > div{{
            background-color: transparent;
        }}
        [data-testid="stChatMessage"]{{
            background-color: transparent;
    
        }}
        [data-testid=stSidebar] {{
            background-color: transparent;

        }}
        header[data-testid="stHeader"]{{
            background-image: url(data:image/jpeg;base64,{encoded_string.decode()});
            background-repeat: no-repeat;
            background-size: cover;
            height: 18%;
        }}
        
        section[data-testid="stSidebar"] {{
            top: 16%; 
        }}
        </style>""",
        unsafe_allow_html=True,
    )


    add_background("./assets/app_bg.png")

    st.markdown("<h3 style='text-align: center; color: black;'>Knowledge Assistant</h3>", unsafe_allow_html=True)



    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4()) 

    add_background("nano_bg.png")
    # st.title("Chat to database")
    st.markdown("<h3 style='text-align: center; color: black;'>Analytics Assistant</h3>", unsafe_allow_html=True)
    st.session_state.example_query_selector = get_example_query()


    if "example_query_selector" not in st.session_state:
        st.session_state.example_query_selector = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask the question to talk to database"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], go.Figure):
                st.plotly_chart(message["content"], theme="streamlit", use_container_width=True)
            else:
                st.write(message["content"])
    
    with st.sidebar:

        options = ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
        selected_option = st.selectbox("Select Gemini Model:", options, index= 0)

        
    # Main content area for displaying chat messages
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    # st.sidebar.button('Clear VectorDB', on_click=clear_vector_db)

    user_question = st.chat_input("Ask a question...")
    if user_question != None:
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        # with st.chat_message("user"):
        #     st.write(user_question)

    user_question = st.session_state.messages[-1]["content"]

    if (user_question != None) and st.session_state.example_query_selector:
        # st.chat_message("user").markdown(user_question)
        # st.session_state.messages.append({"role": "user", "content": user_question})

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # response,source_query = get_conversationchain(user_question,selected_option,st.session_state["example_query_selector"])
                    is_viz,kind_chart,title = check_viz(user_question,selected_option,st.session_state["example_query_selector"])
                    if selected_option == "gemini-1.5-pro-latest":
                        model = "models/gemini-1.5-pro-latest"
                    else:
                        model = "models/gemini-1.5-flash"
                    
                    viz_agent = VizAgent(model,st.session_state["example_query_selector"])

                    try:
                        if is_viz == False:
                            response,source_query = get_conversationchain(user_question,selected_option,st.session_state["example_query_selector"])
                            response += "\n\n Query Logs:  \n" + f"```{source_query}```"
                            placeholder = st.empty()
                            # st.chat_message("assistant").markdown(response)
                            placeholder.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

                        else:
                            if kind_chart == "line":
                                cleaned_input = viz_agent.remove_visualization_words(user_question)
                                result_df,kind_chart,title,x_axis,y_axis,color,source_query = viz_agent.combined_agent_executor(cleaned_input,kind_chart,title)
                                if color == "None":
                                    fig = px.line(result_df, x=x_axis, y=y_axis,color_discrete_sequence=["green"],title=title.upper())
                                    # fig.update_traces(marker_color='red')
                                    # fig.show()
                                    response = st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                                    
                                    # st.chat_message("assistant").markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    placeholder = st.empty()
                                    # st.chat_message("assistant").markdown(response)
                                    placeholder.markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    st.session_state.messages.append({"role": "assistant", "content":fig})

                                    # fig.show()
                                else:
                                    fig = px.line(result_df, x=x_axis, y=y_axis,color=color,title=title.upper())
                                    # fig.update_traces(marker_color='red')
                                    response = st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                                    
                                    # st.chat_message("assistant").markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    placeholder = st.empty()
                                    # st.chat_message("assistant").markdown(response)
                                    placeholder.markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    st.session_state.messages.append({"role": "assistant", "content":fig})
                            elif kind_chart == "bar":
                                cleaned_input = viz_agent.remove_visualization_words(user_question)
                                result_df,kind_chart,title,x_axis,y_axis,color,source_query = viz_agent.combined_agent_executor(cleaned_input,kind_chart,title)
                                if color == "None":
                                    fig = px.bar(result_df, x=x_axis, y=y_axis,color_discrete_sequence=["green"],title=title.upper())
                                    response = st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                                    
                                    # st.chat_message("assistant").markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    placeholder = st.empty()
                                    # st.chat_message("assistant").markdown(response)
                                    placeholder.markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    st.session_state.messages.append({"role": "assistant", "content":fig})

                                else:
                                    fig = px.bar(result_df, x=x_axis, y=y_axis,color=color,title=title.upper())
                                    response = st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                                    
                                    # st.chat_message("assistant").markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    placeholder = st.empty()
                                    # st.chat_message("assistant").markdown(response)
                                    placeholder.markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    st.session_state.messages.append({"role": "assistant", "content":fig})

                            elif kind_chart == "scatter":
                                cleaned_input = viz_agent.remove_visualization_words(user_question)
                                result_df,kind_chart,title,x_axis,y_axis,color,source_query = viz_agent.combined_agent_executor(cleaned_input,kind_chart,title)
                                if color == "None":
                                    fig = px.scatter(result_df, x=x_axis, y=y_axis,color_discrete_sequence=["green"],title=title.upper())
                                    response = st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                                    
                                    # st.chat_message("assistant").markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    placeholder = st.empty()
                                    # st.chat_message("assistant").markdown(response)
                                    placeholder.markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    st.session_state.messages.append({"role": "assistant", "content":fig})


                                else:
                                    fig = px.scatter(result_df, x=x_axis, y=y_axis,color=color,title=title.upper())
                                    response = st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                                    
                                    # st.chat_message("assistant").markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    placeholder = st.empty()
                                    # st.chat_message("assistant").markdown(response)
                                    placeholder.markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                    st.session_state.messages.append({"role": "assistant", "content":fig})

                            elif kind_chart == "pie":
                                cleaned_input = viz_agent.remove_visualization_words(user_question)
                                result_df,kind_chart,title,x_axis,y_axis,color,source_query = viz_agent.combined_agent_executor(cleaned_input,kind_chart,title)
                                fig = px.pie(result_df, names=x_axis, values=y_axis,color=x_axis,title=title.upper())
                                response = st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                                
                                # st.chat_message("assistant").markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                placeholder = st.empty()
                                    # st.chat_message("assistant").markdown(response)
                                placeholder.markdown("\n\n Query Logs:  \n" + f"```'{source_query}'```")
                                st.session_state.messages.append({"role": "assistant", "content":fig})
                    except ServiceUnavailable as e:
                        response = "Mohon maaf bot sedang overloaded, silahkan coba lagi nanti"
                        # st.chat_message("assistant").markdown(response)
                        placeholder = st.empty()
                        placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except:
                        try:
                            response,source_query = get_conversationchain(user_question,selected_option,st.session_state["example_query_selector"])
                            response += "\n\n Query Logs:  \n" + f"```'{source_query}'```"
                            # st.chat_message("assistant").markdown(response)
                            placeholder = st.empty()
                            placeholder.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except:
                            response = "Mohon maaf saya tidak bisa menjawab pertanyaan Anda, silahkan cek kembali pertanyaan Anda bisa"
                            # st.chat_message("assistant").markdown(response)
                            placeholder = st.empty()
                            placeholder.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

    full_response = st.session_state.messages[-1]['content']
    # Feedback submission form
    feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                key=st.session_state.session_id,
                on_submit=handle_feedback,
                kwargs={"result": full_response},
            )

    footer()


if __name__ == '__main__':
    main()