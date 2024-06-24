import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()

def initialize_vectorstore(pdf_path, embeddings):
    if not os.path.exists(pdf_path + "/faiss_index_react"):
        print("Creating FAISS index")
        loader = PyPDFLoader(file_path=pdf_path + "/2210.03629v3.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)

        # vectordb at local machine
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index_react")
        print("FAISS index created")
    return FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

def main():
    embeddings = OpenAIEmbeddings()
    current_file_path = os.path.realpath(__file__)
    pdf_path = os.path.dirname(current_file_path)
    vectorstore = initialize_vectorstore(pdf_path, embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # memory = {}

    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        res = retrieval_chain.invoke({"input": user_input})
        answer = res["answer"]
        print(answer)
        
        # Storing context in memory
        # memory[user_input] = answer

if __name__ == "__main__":
    main()
