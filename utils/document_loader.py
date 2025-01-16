import os
import requests
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader

def load_documents(urls):
    docs = []

    for url in urls:
        try:
            if url.endswith(".pdf"):  # Check if it's a PDF
                print(f"Processing PDF: {url}")
                # Download the PDF
                response = requests.get(url)
                file_name = os.path.basename(url)
                with open(file_name, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                # Load the PDF content
                loader = PyPDFLoader(file_name)
                docs.extend(loader.load())
                # Remove the downloaded file after processing
                os.remove(file_name)
            else:  # Assume it's an article
                print(f"Processing Article: {url}")
                loader = UnstructuredURLLoader(urls=[url])
                docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    return docs