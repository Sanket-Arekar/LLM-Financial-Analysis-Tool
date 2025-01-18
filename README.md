# LLM-Financial-Analysis-Tool

The **Financial Analysis Tool** is an advanced application designed to provide financial analysts with actionable insights. Leveraging state-of-the-art AI and natural language processing (NLP) frameworks, this tool streamlines the extraction, analysis, and retrieval of financial data from various sources, making it an essential asset for investment decision-making.

---

## **Features**

- **Document Processing:**
  - Load data from PDFs and URLs for seamless integration.
  - Extract unstructured and structured financial information efficiently.

- **Natural Language Q&A:**
  - Ask specific financial questions and receive accurate answers with source references.

- **Semantic Search:**
  - Retrieve relevant information based on vector embeddings for insightful analysis.

- **Interactive Interface:**
  - Utilize a user-friendly Streamlit interface for uploading files, submitting URLs, and posing queries.

---

## **Architecture**

### **1. Input Layer**
- **Supported Formats:** URLs, PDF files.
- **Document Loaders:**
  - `PyPDFLoader` for PDF extraction.
  - `UnstructuredURLLoader` for parsing online resources.

### **2. Preprocessing**
- **Text Splitting:**
  - Content is divided into manageable chunks using `RecursiveCharacterTextSplitter`.

### **3. Embedding Generation**
- **Embedding Model:** OpenAI’s Embeddings API.
- **Semantic Representation:** Converts text chunks into vector embeddings for similarity search.

### **4. Vector Storage**
- **Database:** FAISS (Facebook AI Similarity Search).
- **Purpose:** Enables fast, scalable search for vectorized content.

### **5. Query Processing**
- **Chains:**
  - `RetrievalQAWithSourcesChain`: Fetches answers with source references.
  - `RetrievalQA`: Handles general Q&A tasks.

### **6. User Interaction**
- **Frontend:** Streamlit-powered interactive dashboard.
- **Backend:** OpenAI’s robust language processing models.

---

## **Processes**

1. **Document Upload or URL Submission:**
   - Users upload files or submit URLs for analysis.

2. **Content Preprocessing:**
   - Documents are split into chunks for efficient processing.

3. **Vector Embedding and Storage:**
   - Each chunk is converted to a vector and stored in the FAISS database.

4. **Query Execution:**
   - Users enter questions, which are processed against the stored vectors.

5. **Response Generation:**
   - The system retrieves the most relevant chunks and generates an answer with source references.

6. **Output Display:**
   - Results are displayed interactively, including visual highlights and citations.

---

## **Technologies Used**
![image](https://github.com/user-attachments/assets/0dc7a2dd-2ff0-4035-b893-ba5cc88d806d)![image](https://github.com/user-attachments/assets/796c73fa-5fa6-419a-8757-724076e45995)![image](https://github.com/user-attachments/assets/698e4bb8-d515-4be2-bb69-6d21d8ac7f3d)![image](https://github.com/user-attachments/assets/ec592248-5805-427c-b2e9-8f0141009b08)![image](https://github.com/user-attachments/assets/ad76b5a9-089e-49a1-ab49-5fba97ad5675)![image](https://github.com/user-attachments/assets/fba1ba56-3984-41c5-ae4f-bd47d1b7c245)



- **Programming Language:** Python
- **Frontend Framework:** Streamlit
- **AI and NLP Frameworks:**
  - LangChain
  - OpenAI API
- **Vector Database:** FAISS
- **Libraries:**
  - `nltk`, `requests`, `unstructured`, `pypdf`
- **Environment Management:** `.env` for API keys

---

## **Installation**

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   - Add your OpenAI API key to the `.env` file:
     ```env
     OPENAI_API_KEY='your_openai_api_key'
     ```

4. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

---

## **Usage Examples**

### **Example 1: Financial Document Analysis**
1. Upload a PDF containing financial statements.
2. Enter a query such as:
   ```
   What is the net revenue for Q3?
   ```
3. View the response:
   - **Answer:** $10 million

### **Example 2: URL-Based Insights**
1. Submit a URL containing an earnings report.
2. Query:
   ```
   What are the key takeaways from the executive summary?
   ```
3. Receive a concise summary with direct quotes.

---

## **Screenshots**

### **1. Home Page**
![image](https://github.com/user-attachments/assets/9f2d6534-9062-4513-b569-3a2bcbdab1a4)

### **2. Query Results**
![image](https://github.com/user-attachments/assets/f068960d-5bfd-4bfd-a747-a80efc09f5ba)

### **4. Summarized Insights**
![image](https://github.com/user-attachments/assets/8422d911-89c3-4de9-bf7b-73dcc1545589)


---

## **Future Enhancements**

- **Expanded File Format Support:**
  - Include Excel and Word documents.

- **Advanced Analytics:**
  - Add predictive modeling and financial forecasting.

- **Integration:**
  - Support for additional APIs like Bloomberg and Reuters.
  - Integrating Agents to use external tools

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Submit a pull request.

---
