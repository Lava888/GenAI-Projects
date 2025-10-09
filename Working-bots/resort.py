from langchain_openai import ChatOpenAI
import httpx
from openai import OpenAI
from IPython.display import Image
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,END, StateGraph
from dataclasses import dataclass
import streamlit as st
 
def get_eli_chat_model(temperature: float = 0.0, model_name: str = "azure_ai/genailab-maas-DeepSeek-V3-0324"):
    # Create an instance of the OpenAI client
    client = OpenAI(
        api_key="sk-kFX3rQYe-dHtSEbfudJyHg",
        base_url="https://genailab.tcs.in",
        http_client=httpx.Client(verify=False),
    )
    # Create an instance of ChatOpenAI
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="sk-kFX3rQYe-dHtSEbfudJyHg",
        base_url="https://genailab.tcs.in",
    )
    # Now we plug the OpenAI client into our langchain-openai interface
    llm.client = client.chat.completions
    return llm
 
model = get_eli_chat_model()
 
@dataclass
class ResortState:
    email: str
    classification: str =""
    reply: str=""
 
def classify_email(state: ResortState):
    email_content = state.email
    prompt = (
        "You are an expert in classifying emails based on their content.\n"
        "Classify the email below as either 'enquiry' or 'feedback'.\n"
        "Strictly return either 'enquiry' or 'feedback'.\n\n"
        f"{email_content}"
    )
    response = model.invoke(prompt)
    return {"classification": response}
 
def reply_enquiry(state: ResortState):
    email_content = state.email
    prompt = f"Reply to the following enquiry:\n\n{email_content}"
    reply = model.invoke(prompt)
    return {"reply": reply}
 
def reply_feedback(state: ResortState):
    email_content = state.email
    prompt = f"Reply to the following feedback:\n\n{email_content}"
    reply = model.invoke(prompt)
    return {"reply": reply}
 
def route(state: ResortState):
    return "reply_enquiry" if state.classification == "enquiry" else "feedback"
 
# Define the workflow
workflow = StateGraph(ResortState)
workflow.add_node("classify_email", classify_email)
workflow.add_node("reply_enquiry", reply_enquiry)
workflow.add_node("reply_feedback", reply_feedback)
 
 
workflow.add_edge(START, "classify_email")
workflow.add_conditional_edges("classify_email", route,
                               {"enquiry": "reply_enquiry",
                                "feedback": "reply_feedback",
                                })
workflow.add_edge("reply_enquiry", END)
workflow.add_edge("reply_feedback", END)
 
# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
# Image(app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0))
 
config = {
        "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": "1",
         }
    }
 
email = st.chat_input("Enter your email:")
if email:
    with st.chat_message("user"):
        st.write(email)
    with st.spinner("Processing..."):
        inputs = {"email": email}
        print("Processing....")
        final_state = app.invoke(inputs, config)
        print(final_state['email'])
        print("Classiifcation:",final_state['classification'].content)
        print("Reply Mail:==>", final_state['reply'].content)
    with st.chat_message("assistant"):
        st.write(final_state['reply'].content)
 
 
 