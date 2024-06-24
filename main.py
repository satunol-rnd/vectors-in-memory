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

if __name__ == "__main__":
    print("hi")
    embeddings = OpenAIEmbeddings()
    # get current python file path
    current_file_path = os.path.realpath(__file__)

    pdf_path = os.path.dirname(current_file_path)
    # check if faiss_index_react exists
    if not os.path.exists(pdf_path + "/faiss_index_react"):
        print("creating faiss index")
        loader = PyPDFLoader(file_path=pdf_path + "/2210.03629v3.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)

        # vectordb at local machine
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index_react")
        print("faiss index created")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # print("prompt : ", retrieval_qa_chat_prompt)
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    # res = retrieval_chain.invoke({"input": "what is the context about?"})
    # res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    # res = retrieval_chain.invoke({"input": "Give me conclusion about context"})
    res = retrieval_chain.invoke({"input": "Explain ReAct"})
    print(res["answer"])
