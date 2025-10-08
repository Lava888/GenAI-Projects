import sqlite3
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain.globals import set_verbose, set_debug
from typing import Dict, Optional
import uuid
from langchain_openai import ChatOpenAI
import httpx
from openai import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel
import re
import sqlite3
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
import streamlit as st



# Connect to (or create) the database
conn = sqlite3.connect('medicine_inventory.db')
cursor = conn.cursor()

# Create the medicine table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS medicine (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        manufacture_date DATE NOT NULL,
        expiry_date DATE NOT NULL,
        stock INTEGER NOT NULL,
        price REAL NOT NULL
    )
''')

# Commit and close
conn.commit()
conn.close()

# Connect to the database
conn = sqlite3.connect('medicine_inventory.db')
cursor = conn.cursor()

# Insert sample medicines
sample_medicines = [
    ('Paracetamol', '500mg Paracetamol', '2025-01-15', '2027-01-15', 100, 1.50),
    ('Ibuprofen', '200mg Ibuprofen', '2024-12-01', '2026-12-01', 150, 2.00),
    ('Amoxicillin', '250mg Amoxicillin', '2025-03-10', '2026-09-10', 80, 3.75),
    ('Cetirizine', '10mg Cetirizine', '2025-05-20', '2027-05-20', 200, 1.20),
    ('Metformin', '500mg Metformin', '2025-02-28', '2028-02-28', 120, 2.50)
]

cursor.executemany('''
    INSERT INTO medicine (name, content, manufacture_date, expiry_date, stock, price)
    VALUES (?, ?, ?, ?, ?, ?)
''', sample_medicines)

# Commit and close
conn.commit()
conn.close()


db = SQLDatabase.from_uri("sqlite:///medicine_inventory.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM medicine LIMIT 10;")


set_verbose(True)
set_debug(True)

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

client = httpx.Client(verify=False)
# Create an instance of ChatOpenAI
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model = "azure/genailab-maas-gpt-4o",
    api_key="sk-kFX3rQYe-dHtSEbfudJyHg", # Will be provided during event. And this key is for 
    http_client = client
)


template = '''
Given an input question, create a syntactically correct {dialect} query to run to help find the answer.
Unless the user specifies in his question a specific number of examples they wish to obtain,
always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which tables.
Only use the following tables:
{table_info}
Question: {input}
'''
system_message_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['dialect', 'input', 'table_info', 'top_k'],
        template=template
    )
)
query_prompt_template  = ChatPromptTemplate(
    messages=[system_message_prompt]
)

class SQLState(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(BaseModel):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: SQLState):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    #print(type(result))
    output = re.split("query=",str(result))
    #return(output[1])
    sql_query = output[1]
    if sql_query.startswith('"') and sql_query.endswith('"'):
        sql_query = sql_query[1:-1]
    #return sql_query
    return {"query": sql_query}

def execute_query(state: SQLState):
    """Execute SQL query."""
    conn = sqlite3.connect('medicine_inventory.db')
    cursor = conn.cursor()
    query = state["query"].strip("'\"")
    # Execute the SELECT query
    cursor.execute(query)
    # Fetch the result
    result = cursor.fetchall()
    conn.close()
    print(result)
    return {"result": str(result)}

def generate_answer(state: SQLState):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.From SQL query result summarize the details and explain in details.Avoid explaining about SQL query. Do not mention anything about SQL while generating answer.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    
    response = llm.invoke(prompt)
    return {"answer": response.content}


sql_workflow = StateGraph(SQLState)
sql_workflow.add_node("write_query", write_query)
sql_workflow.add_node("execute_query", execute_query)
sql_workflow.add_node("generate_answer", generate_answer)

sql_workflow.add_edge(START, "write_query")
sql_workflow.add_edge("write_query","execute_query")
sql_workflow.add_edge( "execute_query","generate_answer")
sql_workflow.add_edge("generate_answer", END)

sql_memory = MemorySaver()
sql_graph = sql_workflow.compile(checkpointer=sql_memory)
print("SQL GRAPH DONE")
sql_config = {"configurable": {"thread_id": "1"}}

user_query = st.chat_input("ask your query here")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    query = {"question": user_query}
    for step in sql_graph.stream(query, sql_config, stream_mode="updates"):
        print(step)
    
    response = step['generate_answer']['answer']
    with st.chat_message("assistant"):
        st.write(response)   

