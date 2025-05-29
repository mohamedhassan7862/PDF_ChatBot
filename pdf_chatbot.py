# pdf_chatbot.py
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FLAN_T5_MODEL = "google/flan-t5-large"

class PDFChatbot:
    def __init__(self, pdf_path: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = self._load_llm_pipeline()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = self._initialize_chain(pdf_path)

    def _load_llm_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_T5_MODEL)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        return HuggingFacePipeline(pipeline=pipe)

    def _initialize_chain(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory
        )

    def ask(self, question: str) -> str:
        return self.qa_chain.run(question=question)
