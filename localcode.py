from langchain_community import document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_community import embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.docstore.document import Document

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings


template = """ please answer the question according to the context given, if answer is not related to the document then say you don't know
Context: {context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        formatted_content = doc.page_content.replace("\n", " ").strip()
        formatted_docs.append(formatted_content)
    return "\n\n".join(formatted_docs)

client = MilvusClient(uri="http://localhost:19530")
print("connection to milvus:success")
loader = PyPDFDirectoryLoader("./docs/")
data = loader.load()

# text_data=format_docs(data)
# doc = Document(page_content=text_data)
# documents = [Document(page_content=str(content)) for content in doc]

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

# print(all_splits)

vectorstore = Milvus.from_documents(documents=all_splits, embedding=embeddings.ollama.OllamaEmbeddings(model='all-minilm'), collection_name="testing") #test
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# prompt = hub.pull("rlm/rag-prompt")
# mxbai-embed-large	334M
# nomic-embed-text	137M
# all-minilm	23M


# Wrap the text_data string inside a Document object
# doc = Document(page_content=text_data)

# # Pass the Document object to split_documents()
# all_splits = text_splitter.split_documents([doc])


while (True):    
    question=input("input the question to feed-->")
    
    llm = Ollama(
    model="llama3",
    stop=["<|eot_id|>"],
    num_predict=200
    )
    # below two lines enable in debug mode
    docs = retriever.get_relevant_documents(question)
    print(docs)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )

    print("llama3 answer-->")
    output_string = ""
    for chunk in rag_chain.stream(question):
        output_string += chunk
    print(output_string)
    print("----------------<>----------------")



    # below two lines enable in debug mode
    # docs = retriever.get_relevant_documents(question)
    # print(docs[0].page_content)
    llm = Ollama(
    model="mistral",
    stop=["<|eot_id|>"],
    num_predict=200
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )

    print("mistral answer-->")
    output_string = ""
    for chunk in rag_chain.stream(question):
        output_string += chunk
    print(output_string)
    print("----------------<>----------------")

