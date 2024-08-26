from langchain_community.utilities import SQLDatabase
import os
from langchain.chains import create_sql_query_chain
from operator import itemgetter
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.agents import create_tool_calling_agent
from dotenv import load_dotenv
from langchain.agents.agent import AgentExecutor
import streamlit as st
from sqlalchemy import create_engine
import plotly.express as px
import pandas as pd



load_dotenv()
class VizAgent:
    def __init__(self, model_usage,example_query_selector) -> None:
        # Initialize Gemini Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
            # google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize Gemini Chat model
        self.model = ChatGoogleGenerativeAI(
            # model="models/gemini-1.5-pro-latest",
            model= str(model_usage),
            temperature=0,
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            # google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

        self.example_query_selector = example_query_selector
        # self.mysql_uri = 'mysql+mysqlconnector://root:1234@localhost:3306/db_cc'
        # self.sqlite_uri = 'sqlite:///feedback.db'

        # self.mysql_uri = f'mysql://avnadmin:{os.getenv("DB_PASSWORD")}@mysql-14927681-techconnect.h.aivencloud.com:13145/defaultdb'
        self.mysql_uri = f'mysql://avnadmin:{st.secrets["DB_PASSWORD"]}@mysql-14927681-techconnect.h.aivencloud.com:13145/defaultdb'

        self.db = SQLDatabase.from_uri(self.mysql_uri,
                                include_tables=['dummy_trx'], # including only one table for illustration
                                sample_rows_in_table_info=10)
        
    def remove_visualization_words(self,user_input):
        # Prepare the input for the agent
        prompt_input = {"input": user_input}

        # Define the prompt template to remove visualization words from user input
        remove_visualization_words_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an intelligent assistant."),
            HumanMessagePromptTemplate.from_template(
                """
                The user has requested data visualization. Please remove any words or phrases that refer to visualization 
                (e.g., 'chart', 'plot', 'graph', 'visualize', etc.) from the following input, so it can be used as a SQL query.

                User input: "{input}"
                
                Cleaned Input:
                """
            ),MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Create an agent to run the prompt and get the cleaned input
        clean_input_agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(self.model, [], remove_visualization_words_prompt),
            tools=[],  # No tools needed for this step, we're just using the prompt
            verbose=True,
        )

        # Run the agent with the user input
        clean_input_result = clean_input_agent_executor(prompt_input)

        # Extract the cleaned input
        cleaned_input = clean_input_result['output'].strip()

        return cleaned_input

    def check_viz_or_not(self,query):
        visualization_check_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are an intelligent assistant."),
        HumanMessagePromptTemplate.from_template(
            """
            Your task is to determine whether the user wants a data visualization explicitly in the text or just a SQL query result.
            Please analyze the following user input whether they explicitly want visualization or not and respond with either "visualization" if the user wants a visualization,
            or "query" if the user only wants the data.
            If you identify user wants "visualization". Please analyze what kind of visualization and respond with either "line chart" if the user wants a line chart, "bar chart" if the user wants bar chart,"table" if the user wants table, "scatter plot" if the user wants scatter plot, or "pie chart" if the user wants pie chart
            If there is neither words of "visualization", "line chart", "bar chart", "pie chart", nor "scatter plot" please give response "query"
            Please give chart title refer to user input.
            I want your response could be the following:
            kind chart:
            title:
            User input: "{input}"
            
            Response:
            """
        ),MessagesPlaceholder("agent_scratchpad")
    ])
        # Prepare the input for the agent
        prompt_input = {"input": query,"agent_scratchpad": []}
        
        # Create an agent to run the prompt and get the response
        intent_agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(self.model, [], visualization_check_prompt),
            tools=[],  # No tools needed for this step, we're just using the prompt
            verbose=True,
        )

        # Run the agent with the user input
        intent_result = intent_agent_executor(prompt_input)

        # Determine if the response indicates a visualization is needed
        intent_text = intent_result['output'].strip().lower()

        if ("line chart" in intent_text) | ("bar chart" in intent_text) | ("pie chart" in intent_text) | ("scatter plot" in intent_text):
            print("============")
            print(self.remove_visualization_words(query))
            print("============")

            if "doesn't contain any words or phrases related to visualization" in self.remove_visualization_words(query):
                return False,"NULL","NULL"
            else:
                if "line chart" in intent_text:
                    kind_chart = "line"
                    get_title = intent_text.split("title:")[-1]
   

                    return True,kind_chart,get_title
                
                elif "bar chart" in intent_text:
                    kind_chart = "bar"
                    get_title = intent_text.split("title:")[-1]

                    return True,kind_chart,get_title
                elif "pie chart" in intent_text:
                    kind_chart = "pie"
                    get_title = intent_text.split("title:")[-1]

                    return True,kind_chart,get_title

                elif "scatter plot" in intent_text:
                    kind_chart = "scatter"
                    get_title = intent_text.split("title:")[-1]

                    return True,kind_chart,get_title
                elif "table" in intent_text:
                    return False,"NULL","NULL"
                            
        else:
            return False,"NULL","NULL"


    def map_columns_to_plotly(self,df):
        # Extract the column names from the DataFrame
        plotly_mapping_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are an intelligent assistant."),
        HumanMessagePromptTemplate.from_template(
            """
            You have been provided with a DataFrame that was generated from a SQL query. The DataFrame contains the following columns: {columns}.
            
            Your task is to map these columns to the appropriate parameters for creating a Plotly chart. Specifically, you need to suggest columns for the following parameters:
            - x: The column that should represent the x-axis (usually a time-related or categorical column).
            - y: The column that should represent the y-axis (usually a numerical column).
            - color: The column that should determine the color of different elements in the chart (optional, typically categorical).
            - size: The column that should determine the size of elements in a scatter plot or bubble chart (optional, typically numerical).
            - hover_data: The columns that should be displayed when hovering over data points in the chart (optional).

            Please provide your recommendations in the following format:
            "x: [column_name], y: [column_name], color: [column_name or 'None'], size: [column_name or 'None'], hover_data: [list of column_names or 'None']"

            Example DataFrame columns:
            {columns}

            Response:
            """
        ),MessagesPlaceholder("agent_scratchpad")
    ])
        columns = df.columns.tolist()

        # Prepare the prompt input
        prompt_input = {"columns": ', '.join(columns)}

        # Create an agent to run the prompt and get the mappings
        mapping_agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(self.model, [], plotly_mapping_prompt),
            tools=[],  # No tools needed for this step, we're just using the prompt
            verbose=True,
        )

        # Run the agent with the column names
        mapping_result = mapping_agent_executor(prompt_input)

        # Extract and return the mappings from the LLM's response
        response_text = mapping_result['output'].strip()
        
        # Initialize an empty dictionary for the mappings
        mapping_dict = {}

        # Split the response by commas to get each mapping
        try:
            for part in response_text.split(','):
                if ':' in part:
                    key, value = part.split(':', 1)  # Split only on the first colon
                    mapping_dict[key.strip()] = value.strip()
                else:
                    print(f"Skipping unrecognized format: {part}")
        except Exception as e:
            print(f"Error parsing the response: {e}")

        return mapping_dict



    def combined_agent_executor(self,input_query,kind_chart,title):
        engine = create_engine(self.mysql_uri)
        system_prefix = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" in Bahasa Indonesia as the answer.
        If answering numeric value, use thousand delimiter.
        Make sure using alias column when generating SQL query.
        If month format still %m then convert it to %b


        Here are some examples of user inputs and their corresponding SQL queries:"""

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_query_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input}\nSQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k"],
            prefix=system_prefix,
            suffix="",
        )

        full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=few_shot_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )


        toolkit = SQLDatabaseToolkit(
            db = self.db,
            llm = self.model
        )
        context = toolkit.get_context()
        tools = toolkit.get_tools()

        # Step 1: Run the SQL Query Agent
        query_agent_executor = AgentExecutor(agent=create_tool_calling_agent(self.model, tools, full_prompt), tools=tools, verbose=True,return_intermediate_steps=True)
        output = query_agent_executor({
            "input": input_query,
            "top_k": 5,
            "dialect": "mysql",
            "agent_scratchpad": []})
        print(output)
        
        print(output['intermediate_steps'][-1][0].tool_input['query'])
        
        # Ensure output is captured as a DataFrame
        # result_df = pd.DataFrame(output['output'])
        result_df = pd.read_sql(output['intermediate_steps'][-1][0].tool_input['query'], con=engine)
        # x_axis, y_axis = determine_axes(result_df)
        plotly_parameters = self.map_columns_to_plotly(result_df)
        x_axis =" ".join(map(str, plotly_parameters['x'])).strip() if isinstance(plotly_parameters['x'], tuple) else plotly_parameters['x'] 
        y_axis =" ".join(map(str, plotly_parameters['y'])).strip() if isinstance(plotly_parameters['y'], tuple) else plotly_parameters['y'] 
        color=" ".join(map(str, plotly_parameters.get('color', None))).strip() if isinstance(plotly_parameters.get('color', None), tuple) else plotly_parameters.get('color', None) 

        source_query = output['intermediate_steps'][-1][0].tool_input['query']
        return result_df,kind_chart,title,x_axis,y_axis,color,source_query

