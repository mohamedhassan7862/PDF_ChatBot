import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FLAN_T5_MODEL = "google/flan-t5-base"
TEMP_DIR = "./temp_pdfs"

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def get_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_T5_MODEL)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=pipe)

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
    st.title("📄🗣️ PDF Chatbot")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if not uploaded_file:
        st.info("Please upload a PDF to start chatting.")
        return

    # Save PDF locally
    os.makedirs(TEMP_DIR, exist_ok=True)
    pdf_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Initialize chain and memory once
    if "qa_chain" not in st.session_state:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        embeddings = get_embedding_model()
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

        llm = get_llm_pipeline()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
        st.session_state.qa_chain = qa_chain
        st.session_state.memory = memory

    # Display chat history in a scrollable container
    chat_container = st.container()
    with chat_container:
        if hasattr(st.session_state.memory, 'chat_memory'):
            for msg in st.session_state.memory.chat_memory.messages:
                role = msg.type.lower()
                if role == 'human':
                    with st.chat_message("user"):
                        st.markdown(msg.content)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg.content)

    # Always show the input at the bottom via chat_input
    query = st.chat_input("Type your question...")
    if query:
        response = st.session_state.qa_chain.run(question=query)
        # Display user message and response instantly
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
