import streamlit as st
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
#from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

ollama_url = "https://diagnoflow.loca.lt/"
ollama_model = "gemma:2b"

template = """you are an ai chatbot having a conversation with a human. Also you can markdown content during conversation.


{history}
human: {human_input}
ai: """

prompt_template = PromptTemplate(input_variables = ["history","human_input"], template=template)

msgs = StreamlitChatMessageHistory(key="follow_up_bot_key")
memory = ConversationBufferMemory(memory_key="history",chat_memory=msgs)

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

llm = Ollama(model=ollama_model,base_url=ollama_url,callback_manager = CallbackManager([StreamingStdOutCallbackHandler]))
llm_chain = LLMChain(llm=llm,prompt=prompt_template,memory=memory)

st.set_page_config(layout='wide')

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in msgs.messages:
    st.chat_message(msg.type).markdown(msg.content)
if prompt:= st.chat_input("Say something: "):
    st.chat_message("user").markdown(prompt)

    response = llm_chain.run(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)