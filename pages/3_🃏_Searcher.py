import streamlit as st
import time
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
try:
    from googlesearch import search
except ImportError: 
    print("No module named 'google' found")
 
load_dotenv()


def get_vectorstore_from_url(urls):
    
    document = None
    
    for url in urls:
        # get the text in document form
        loader = WebBaseLoader(url)
        part_document = loader.load()
        if document == None:
            document = part_document
        else:
            document = document + part_document
    
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    vector_store = None
    for i in range(len(document_chunks)):
        if vector_store is None:
            vector_store = Chroma.from_documents([document_chunks[i]], OpenAIEmbeddings())
        else:
            vector_store.add_documents([document_chunks[i]])
        # print( "########################################################################", i, len(document_chunks))
        progress_bar.progress((i/len(document_chunks)))
        time.sleep(0.05)


    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  

# user input
user_query = st.chat_input("Type your question here...")
if user_query is not None and user_query != "":
    urls = []
    st.write("Websites I'm using to generate my answer:")
    placeholder_table = st.empty()
    display = ""
    for j in search(user_query, tld="co.in", num=5, stop=5, pause=2):
        urls.append(j)
        display = display + "- " + j + " \n"
        placeholder_table.markdown(display)
    progress_bar = st.progress(0)
    placeholder_text = st.empty()
    placeholder_text.write("Making sure to generate the best answer...")
    st.session_state.vector_store = get_vectorstore_from_url(urls)
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    progress_bar.empty()
    placeholder_text.empty()

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
 
