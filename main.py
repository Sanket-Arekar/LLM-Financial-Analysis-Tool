import os
import nltk
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader,PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from utils.document_loader import load_documents

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from dotenv import load_dotenv
load_dotenv()  ##take environment variable from .env

st.title("Financial Analysis Tool")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

go_clicked = st.sidebar.button("Go")

main_placefolder = st.empty()
# Replace `text-davinci-003` with `gpt-3.5-turbo` or `gpt-4`
# Initialize the LLM with OpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4"
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    verbose=False  # Disable verbose logging that might trigger the issue
)

# llm = OpenAI(model="gpt-3.5-turbo",temperature = 0.7, max_tokens = 500)

if go_clicked:
    #Scrape the Data and load
    main_placefolder.text("Data Loading now.....")

    # loader = UnstructuredURLLoader(urls=urls)
    # main_placefolder.text("Data Loading now.....")
    # data = loader.load()

    documents = load_documents(urls)

    # Debug: Check if documents are loaded
    if documents:
        print(f"Successfully loaded {len(documents)} documents.")
        for doc in documents[:3]:  # Print a sample of the content
            print(f"Content: {doc.page_content[:500]}")  # Show first 500 characters
    else:
        print("No documents loaded. Check the URLs or loaders.")


    main_placefolder.text("Splitting docs now.....")
    #Split the data using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","," "],
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)

    main_placefolder.text("Creating Embeddings.....")
    #Create Embeddings for chunks we created
    embeddings = OpenAIEmbeddings()

    # check if docs are loaded
    if not docs:
        raise ValueError("Documents are empty. Ensure document loading is successful.")
    if not embeddings:
        raise ValueError("Embeddings are empty. Ensure embeddings are being generated.")

    # Each chunk is passed through the embeddings object to generate its vector representation.
    vectorindex_openai = FAISS.from_documents(docs, embeddings)

    main_placefolder.text("Creating pickle file.....")
    # Save the FAISS index along with docstore and index_to_docstore_id
    file_path = "vector_index.pkl"
    vectorindex_openai.save_local(file_path)

query = main_placefolder.text_input("Question: ")
if query:
    file_path = "vector_index.pkl"
    embeddings = OpenAIEmbeddings()
    if os.path.exists(file_path):
        # Load the FAISS index
        vectorIndex = FAISS.load_local(
            file_path, embeddings
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorIndex.as_retriever(),
            chain_type="refine"
        )
        langchain.debug = True

        # Run the query through the chain
        response = chain({"query": query}, return_only_outputs=True)

        print(response)

        st.header("Answer")
        st.subheader(response["result"])








