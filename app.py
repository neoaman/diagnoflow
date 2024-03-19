import dash_mantine_components as dmc
from dash import Dash, html, Input, Output, State, dcc
import json
from urllib.request import urlretrieve
import dash_auth
from dash_iconify import DashIconify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# BASE_URL = "https://d1agn0fl0w.loca.lt/"
BASE_URL = "https://d1agn0flo.loca.lt/"

try:
    db = FAISS.load_local("assets/data/medical_index",embeddings=OllamaEmbeddings(model="gemma:2b",base_url=BASE_URL),allow_dangerous_deserialization=True)
except Exception as e:
    with open("assets/data/config.json",'r') as cj:
        vector_config = json.load(cj)
    urlretrieve(vector_config["index_faiss_url"],"assets/data/medical_index/index.faiss")
    urlretrieve(vector_config["index_pkl_url"],"assets/data/medical_index/index.pkl")
    db = FAISS.load_local("assets/data/medical_index",embeddings=OllamaEmbeddings(model="gemma:2b",base_url=BASE_URL),allow_dangerous_deserialization=True)


prompt = hub.pull("rlm/rag-prompt")
retriever = db.as_retriever()
llm = Ollama(model="gemma:2b",base_url=BASE_URL,callback_manager = CallbackManager([StreamingStdOutCallbackHandler]))

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

VALID_USERNAME_PASSWORD_PAIRS = [
    ['amanneo', 'P@ssw0rd']
]

app = Dash(__name__)
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )
server = app.server # needed


app.layout = html.Div([
    dcc.Store(id="store_chat_history",data=[("bot","Hello, I am DiagnoFlow assistance, How can I help you ?")]),
    dmc.Title("DiagnoFlow"),
    dmc.Alert(
        "Hi from Aman (creator of DiagnoFlow). We will like to serve you!",
        title="Welcome to DiagnoFlow",
        color="violet",
    ),
    dmc.Affix(
        dmc.Button("DiagnoFlow Assist",id="btn_chatbox",n_clicks=1,color="violet"), position={"bottom": 20, "right": 20}
    ),

    dmc.Affix(
        dmc.Stack(
            [
                html.Div(id="conversationbox"),
                dcc.Input(id="input_query",debounce=True)
            ],className="glass chatbox",align="stretch",justify="flex-end"),
            position={"bottom": 60, "right": 20},
            id="chatbox",
        # style={"display":"none"}
    ),

])

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
        conversation.append(dmc.Group(
            [
                DashIconify(icon=("ic:baseline-adb" if user=="bot" else "ic:baseline-person" ), width=30),
                dmc.Paper(message,className=user)
            ]
        ))
        # conversation.append(html.Div(message,className=user))
    return dmc.Stack(conversation,align="stretch",justify="flex-end")

@app.callback(
    Output("store_chat_history","data"),
    Output("input_query","value"),
    State("store_chat_history","data"),
    Input("input_query","value")
)
def display_conversation(history,query):
    if not query: return history
    history.append(("human",query))
    history.append(("bot",rag_chain.invoke(query)))
    return history,""


if __name__ == '__main__':
    app.run_server(debug=True)

# gunicorn main:server -b 0.0.0.0:8050 --reload