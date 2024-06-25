import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.prompts import PromptTemplate

load_dotenv()

def initialize_vectorstore(pdf_path, embeddings):
    if not os.path.exists(pdf_path + "/faiss_index_react"):
        print("Creating FAISS index")
        # loader = PyPDFLoader(file_path=pdf_path + "/2210.03629v3.pdf")
        loader = PyPDFLoader(file_path=pdf_path + "/BA_SINEC-NMS_76_en-US.pdf")
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
def create_chat_history(memory):
    if len(memory) == 0:
        return "no chat history yet"
    # format key as Q and value as A from 2 last items in memory
    memory = dict(list(memory.items())[-2:])
    l = [f"HUMAN\n{key}\n\nCHATBOT\n{value}" for key, value in memory.items()]
    return "\n\n".join(l)

def main():
    embeddings = OpenAIEmbeddings()
    current_file_path = os.path.realpath(__file__)
    pdf_path = os.path.dirname(current_file_path)
    vectorstore = initialize_vectorstore(pdf_path, embeddings)

    template = """
You are a nice chatbot identified as CHATBOT. 
You are having a conversation with a human.
Answer any use questions based solely on the context below:

<context>
{context}
</context>

Chat History:
{chat_history}

HUMAN
{input}
    """

    retrieval_qa_chat_prompt = PromptTemplate.from_template(template=template).partial(
        chat_history="no chat history yet",
    )

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    memory = {}

    while True:
        print("\n\n---------------------------------------------------------------------")
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        res = retrieval_chain.invoke({"input": user_input, "chat_history": create_chat_history(memory)})
        answer = res["answer"].strip()

        # Storing context in memory, check if answer starts with "SYSTEM" if yes remove it
        if answer.startswith("SYSTEM"):
            answer = answer[6:]
            answer = answer.strip()
        elif answer.startswith("CHATBOT"):
            answer = answer[8:]
            answer = answer.strip()

        memory[user_input] = answer
        print("---------------------------------------------------------------------")
        print("CHATBOT:\n", answer)
        print("----------------chat history:----------------------------------------\n")
        print(res['chat_history'])
        
        

if __name__ == "__main__":
    main()
