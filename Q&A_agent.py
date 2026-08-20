import os
import wget
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# 2. Download & Load the Document
filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

if not os.path.exists(filename):
    wget.download(url, out=filename)
    print('\nFile downloaded')

loader = TextLoader(filename)
documents = loader.load()

# 3. Split the Document into Chunks
# Using 1000 characters as defined in the original project
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"Document split into {len(texts)} chunks.")

# 4. Embed and Store (Open Source)
# This utilizes local compute for vectorization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docsearch = Chroma.from_documents(texts, embeddings)
retriever = docsearch.as_retriever()
print('Document ingested into ChromaDB.')

# 5. Initialize Gemini LLM
# Using Gemini 1.5 Flash for fast, conversational responses
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

# 6. Create History-Aware Retriever
# This rephrases the user's question based on the chat history to ensure the vector search is accurate
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# 7. Create the Q&A Chain (with anti-hallucination prompt)
qa_system_prompt = (
    "Use the information from the retrieved context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, definitely do not try to make up an answer."
    "\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 8. Wrap up and make it an agent
def qa_agent():
    print("\n--- Agent Started ---")
    print("Type 'quit', 'exit', or 'bye' to stop.")
    
    chat_history = []
    
    while True:
        query = input("\nQuestion: ")
        
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
            
        # Execute the chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        answer = result["answer"]
        
        # Append to manual memory list
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer)
        ])
        
        print(f"Answer: {answer}")

if __name__ == "__main__":
    qa_agent()