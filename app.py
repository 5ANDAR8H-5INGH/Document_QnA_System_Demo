import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

#Step 1 : load pdf
loader = PyPDFLoader('AE.pdf')
documents = loader.load()
print("pdf loaded successfully.....")

#Step 2 : Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)
print('Chunks created: ',len(docs))

#Step 3 : create Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)
print("Embedding model loaded")

#Step 4 : Store in Faiss DB
vectorstore = FAISS.from_documents(docs,embedding_model)
print('Vector DataBase Connected')

#Step 5 : Load Gemmini Model
llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash',temerature = 0.3) # temperature is like creativity i.e high temp mean high creativity
print('LLM is loaded')

#Step 6 :Ask Questions
query = input("Search your Query : ")
matched_docs = vectorstore.similarity_search(query) # Retrival step
chain = load_qa_chain(llm,chain_type='stuff')
response = chain.run(input_documents=matched_docs,question=query) #Generation Step
print('Answer : ')
print(response)