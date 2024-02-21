import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_text = PdfReader(pdf)
        for page in pdf_text.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)

    return chunks

def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # create a vectorstore from the chunks
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

################################################################

load_dotenv()

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
        "chat_history": st.session_state.chat_history_pdf,
        "input": user_query
    })
    
    return response['answer']



# app config
st.set_page_config(page_title="Chat with PDFs", page_icon="📜")
st.title("Chat with PDFs")

# sidebar
with st.sidebar:
    st.subheader("Your documents")
    pdf_docs =  st.file_uploader("Upload your documents here...", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing..."):
            ## Getting text from PDF and splitting it into chunks
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)

            ## Creating a vector store
            st.session_state.vector_store = get_vectorstore(text_chunks)
            
            ## Creating a conversation
            st.session_state.conversation = get_conversation(st.session_state.vector_store)



# session state
if "chat_history_pdf" not in st.session_state:
    st.session_state.chat_history_pdf = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history_pdf.append(HumanMessage(content=user_query))
    st.session_state.chat_history_pdf.append(AIMessage(content=response))
    
    

# conversation
for message in st.session_state.chat_history_pdf:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

