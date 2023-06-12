import streamlit as st
import logging

from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import  FAISS 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    


def handle_user_input(user_question):
    """
    Handles user input and generates bot response.

    :param user_question: User's question.
    :type user_question: str
    """
    if not st.session_state.conversation:
        logger.error("Conversation chain not initialized.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def get_pdf_text(pdf_documents):
    rawtext = ""
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            rawtext+=page.extract_text()
    return rawtext

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200, # this makes sure that the chunks overlap a bit so that we don't miss any words
        length_function = len
        )
    chunks = text_splitter.split_text(raw_text)
    return chunks
    
# def get_vectorstore_v2(chunks,embeddings):
#     vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     return vectorstore

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Creates a conversation chain from a vector store.

    :param vectorstore: Vector store.
    :type vectorstore: FAISS
    :return: Conversation chain.
    :rtype: ConversationalRetrievalChain
    """
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
    

def main():
    load_dotenv()
    st.set_page_config(page_title="LLM for PDFs", page_icon="🦍", layout="wide")
    st.header("Too lazy to read all the PDFs? Good news, I am here to help!")
    st.text_input("Ask me anything from the uploaded PDFs!")    

    with st.sidebar:
        st.header("Your uploaded PDFs:")
        pdf_documents = st.file_uploader("Upload your PDFs here. Press 'Submit' once finished.", type="pdf", accept_multiple_files=True, key="pdf_documents")
        st.button("Submit")
        with st.spinner("Loading.."):
            raw_text = get_pdf_text(pdf_documents) # getting the text from the PDFs
            # st.write(raw_text)
            chunks = get_text_chunks(raw_text)# generate text chunks
            st.write(chunks)
            
            vectorstore = get_vectorstore(chunks)
            # oaie=OpenAIEmbeddings()
            # vectorstore = get_vectorstore_v2(chunks,oaie) # creating the db/vectorstore with OpenAI embeddings (check: https://openai.com/pricing)
            # InstructorEmbeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
            # vectorstore = get_vectorstore_v2(chunks,InstructorEmbeddings) # works on your own hardware, therefore slower, but FREE & higher rated: https://huggingface.co/spaces/mteb/leaderboard
            
            st.session_state.conversation = get_conversation_chain(vectorstore) # create the conversation chain

            
if __name__ == '__main__':
    main()
