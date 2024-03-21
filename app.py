import dash_mantine_components as dmc
from dash import Dash, html, Input, Output, State, dcc
import json
from urllib.request import urlretrieve
import dash_auth
from dash_iconify import DashIconify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import format_document

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.llms import Ollama
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# BASE_URL = "https://d1agn0fl0w.loca.lt/"
# BASE_URL = "https://diagnoflow.loca.lt/"
BASE_URL = "https://eager-lines-prove.loca.lt/"
vector_embedding = "gemma:2b"
llm_model = "gemma"


# Download the Vectorstore if unavailable
try:
    db = FAISS.load_local("assets/data/medical_index",embeddings=OllamaEmbeddings(model=vector_embedding,base_url=BASE_URL),allow_dangerous_deserialization=True)
except Exception as e:
    with open("assets/data/config.json",'r') as cj:
        vector_config = json.load(cj)
    urlretrieve(vector_config["index_faiss_url"],"assets/data/medical_index/index.faiss")
    urlretrieve(vector_config["index_pkl_url"],"assets/data/medical_index/index.pkl")
    db = FAISS.load_local("assets/data/medical_index",embeddings=OllamaEmbeddings(model=vector_embedding,base_url=BASE_URL),allow_dangerous_deserialization=True)


# Create the RAG with langchain
prompt = hub.pull("rlm/rag-prompt")
retriever = db.as_retriever()
llm = Ollama(model=llm_model,base_url=BASE_URL,callback_manager = CallbackManager([StreamingStdOutCallbackHandler]))

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



# HACK New approach




DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
answer = {
    "answer": final_inputs | ANSWER_PROMPT | llm,
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer








app = Dash(__name__)
server = app.server # needed for gunicorn 


app.layout = html.Div([
    dcc.Store(id="store_chat_history",data=[("bot","Hello, I am DiagnoFlow assistance, How can I help you ?")]),
    dmc.Title("DiagnoFlow"),
    dmc.Alert(
        "Hi from Neo (creator of DiagnoFlow). We will like to serve you!",
        title="Welcome to DiagnoFlow",
        color="violet",
    ),
    dmc.Affix(
        dmc.Button("DiagnoFlow Assist",id="btn_chatbox",n_clicks=1,color="violet"), position={"bottom": 20, "right": 20}
    ),

    dmc.Affix(
        dmc.Stack(
            [
                dmc.Paper(id="conversationbox"),
                dcc.Input(placeholder="Write your query and hit enter",id="input_query",debounce=True,style={'margin':'10px'})
            ],
            className="glass chatbox",
            align="stretch",justify="flex-end"
            ),
            position={"bottom": 60, "right": 20},
            id="chatbox",
    ),

]
)

@app.callback(
    Output("chatbox","style"),
    Input("btn_chatbox","n_clicks")
)
def toggle_chatbox(btn):
    return ({"display":"none"} if (btn % 2 == 0) else {"display":""})

@app.callback(
    Output("conversationbox","children"),
    Input("store_chat_history","data"),
)
def display_conversation(data):

    conversation = []
    for user,message in data:
        conversation.append(dmc.Paper(message,className=user))
    return dmc.Stack(conversation,align="stretch",justify="flex-end")

@app.callback(
    Output("store_chat_history","data"),
    Output("input_query","value"),
    State("store_chat_history","data"),
    Input("input_query","value"),
    prevent_initial_call=True
)
def display_conversation(history,query):
    if not query: return history
    history.append(("human",query))
    history.append(("bot",rag_chain.invoke(query)))

    # result = final_chain.invoke({"question":query})
    # print(result["answer"])
    # history.append(("bot",result["answer"]))

    # print("="*20)
    # print(result)


    return history,""


if __name__ == '__main__':
    app.run_server(debug=True)

# gunicorn main:server -b 0.0.0.0:8050 --reload