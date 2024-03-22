from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests

from model import get_patient_yaml

st.set_page_config(layout="wide")

if "patient_name" not in st.session_state: 
    st.session_state["patient_name"] = ""
if "patient_info" not in st.session_state: 
    st.session_state["patient_info"] = ""


with st.sidebar:
    base_url=st.text_input("Enter the ollama base url","http://0.0.0.0:11434")
    try:
        model_options = [i["name"] for i in requests.get(f"{base_url}/api/tags").json()["models"]]
    except:
        st.toast(":red[Invalid ollama base url]")
        model_options = [""]
    selected_model = st.selectbox("Select llm model",options=model_options)
    context = st.text_area("Provided context","""
    blood_pressure:
    diastolic: 90 mmHg
    systolic: 135 mmHg
    blood_type: O-
    date_of_birth: '1970-12-30'
    gender: Male
    heart_rate: 72 bpm
    height: 178 cm
    patient_name: Liam Wilson
    reports:
    - date: '2024-03-17'
    description: High Blood Pressure
    findings: Hypertension detected, need for lifestyle modifications
    recommendations:
    - Begin medication as prescribed
    - Reduce salt intake
    - Increase physical activity
    temperature: "36.9 °C"
    weight: 85 kg
    """)

ask_name = st.empty()
with ask_name:
    patient_name = st.text_input("Username")
    st.session_state["patient_name"] = patient_name
    st.session_state["patient_info"] = get_patient_yaml(st.session_state["patient_name"])

if st.session_state["patient_name"]=="":
    st.stop()
else:
    with ask_name:
        st.empty()

initial_instruction = f"""
You are a personalized assistance for patients, based on provided context below in json format, please answer the questions of patient.
Please write name of the parient while greeting and interacting.

Below is one example for you with context and how to handel the questions.

Context:
---------------------------
blood_pressure:
  diastolic: 90 mmHg
  systolic: 135 mmHg
blood_type: O-
date_of_birth: '1970-12-30'
gender: Male
heart_rate: 72 bpm
height: 178 cm
patient_name: Liam Wilson
reports:
- date: '2024-03-17'
  description: High Blood Pressure
  findings: Hypertension detected, need for lifestyle modifications
  recommendations:
  - Begin medication as prescribed
  - Reduce salt intake
  - Increase physical activity
temperature: "36.9 \xB0C"
weight: 85 kg
---------------------------

Question: Hi
Hello Liam How can I help you ?

Question: how are my reports?.
Liam, based on your report doctor identified that you have high blood pressure.

Question: What are the recommendations?
Liam, You need to begin the prescribed medication by doctor, and do some physical activities, please reduce your salt intake.

=============================================================================================================
Now its YOUR TURN to interact with the patient REMEMBER, answer should be short, don't act smart and answer should be specific, always say the patient name in response.

Context:
---------------------------
{st.session_state["patient_info"]}
---------------------------
"""

# print(requests.get(f"{base_url}/api/tags").json())

template = initial_instruction+"""

{history}
Human: {human_input}
AI: """
prompt_template = PromptTemplate(input_variables=["history", "human_input"], template=template)

msgs = StreamlitChatMessageHistory(key="follow_up_bot_key")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message(f"Hi {st.session_state['patient_name']}, How can I help you?")

llm = Ollama(model = selected_model, base_url=base_url,callback_manager = CallbackManager([StreamingStdOutCallbackHandler]))
llm_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)
        

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in msgs.messages:
    st.chat_message(msg.type).markdown(msg.content)

if prompt := st.chat_input("Say something: "):
    st.chat_message("user").markdown(prompt)

    response = llm_chain.run(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)