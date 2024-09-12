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
)
from langchain.agents import create_tool_calling_agent
from dotenv import load_dotenv
from langchain.agents.agent import AgentExecutor
import streamlit as st



load_dotenv()
class SQLAgent:
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


    def generate_response(self, query: str):
        # Write prompt to guide the LLM to generate response
        try:
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
            tools = toolkit.get_tools()

            agent = create_tool_calling_agent(self.model, tools, full_prompt)

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iteration=20,
                return_intermediate_steps=True

                # callbacks=[await asyncio.sleep(5)]
            )

            result = agent_executor.invoke({
                "input": query,
                "top_k": 5,
                "dialect": "mysql",
                "agent_scratchpad": []
            })
            source_query = result['intermediate_steps'][-1][0].tool_input['query']
            # for i in result['intermediate_steps']:
            #     source_query = source_query + i[0].log

            output = result['output']
            if "i don't know" in output.lower():
                return output,"NULL"
            else:
                return output,source_query
        except:
            output = "Oops, maaf tidak bisa jawab pertanyaan ini silahkan parafrase pertanyaan Anda atau ganti pertanyaan yang lainnya"
            source_query = "NULL"
            return output,source_query


