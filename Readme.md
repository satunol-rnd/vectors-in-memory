# Getting Started

```
pipenv install langchain pypdf langchain-openai langchain-community langchainhub faiss-cpu black
```

## ABout This Project

The purpose of this project is to show how to use Langchain with OpenAI and vector database. The program will chop the pdf file into chunks with keys and values. Then it will search for the keys in the vector database and return the values. Then it will feed the values to the LLM by prompting it.