import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

loader = PyPDFLoader("document.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="db")

retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs= {"k": 3})

llm = ChatOpenAI()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever = retriever)

print("Chat started (type 'exit' to quit)\n")

while True:
    query = input("You:")
    if query == "exit":
        break
    result = qa_chain({"query": query})
    print("AI:", result["result"])