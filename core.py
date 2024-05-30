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

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.document_loaders import PyPDFDirectoryLoader



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)    


def process(question):
    #connecting milvus
    client = MilvusClient(uri="http://localhost:19530")
    print("connection to milvus:success")
    loader = PyPDFDirectoryLoader("./docs/")

    data = loader.load()


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    all_splits = text_splitter.split_documents(data)

    vectorstore = Milvus.from_documents(documents=all_splits, embedding=embeddings, collection_name="demo") #test
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    prompt = hub.pull("rlm/rag-prompt")
    llm = Ollama(
        model="llama3",
        stop=["<|eot_id|>"],
        )
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    output_string = ""
    for chunk in rag_chain.stream(question):
        output_string += chunk
    
    return output_string
    




# if client.has_collection('test'): 
#     client.drop_collection('test')


# #loading pdf form url
# print("milvus is connected")    
# PyPDFLoader = document_loaders.PyPDFLoader
# loader = PyPDFLoader(
#     "https://www.nitk.ac.in/document/attachments/3450/Information_Bulletin_%28MBA_2022-24%29.pdf"
# )
# data = loader.load()






# from transformers import AutoTokenizer

# llm = Ollama(
#     model="llama3",  # Replace with the specific model name if needed
#     stop=["<|eot_id|>"],  # Stop token for generation
#     tokenizer=AutoTokenizer.from_pretrained("google/llama-3b"),  # Load tokenizer
#     max_new_tokens=256,  # Maximum number of tokens to generate
#     eos_token_id=terminators,  # End-of-sentence token ID
#     do_sample=True,  # Enable sampling for diverse generation
#     temperature=0.6,  # Controls randomness of the generated text
#     top_p=0.9,  # Filters out low probability tokens
# )








# for custom printing
# for chunks in llm.stream(text):
#     print(chunks, end="", flush=True)


# -->check if the model is using external learned data
# --> check if it remembers the conversation history



# from langchain.memory import ChatMessageHistory

# demo=ChatMessageHistory()

# User: Hi Llama, can you tell me about the history of the internet?
# Llama 3: Sure! The internet has a long and fascinating history. It began as a network of computers developed by the US Department of Defense in the 1960s. Over time, it grew and evolved, becoming the global network we know today.

# User: That's interesting! What were some of the key milestones in its development?

# Llama 3: Some key milestones include:

# 1969: The first ARPANET message is sent between two computers at UCLA and Stanford.
# 1983: TCP/IP protocol is adopted, which becomes the foundation of the modern internet.
# 1991: The World Wide Web is invented by Tim Berners-Lee.
# 2000s: The internet experiences rapid growth and becomes an essential part of daily life.

# User: Wow, that's a lot of progress in a relatively short time. What are some of the biggest challenges facing the internet today?

# Llama 3: Some of the biggest challenges include:

# Cybersecurity threats: Protecting against hackers and malware.
# Digital divide: Ensuring everyone has access to the internet.
# Misinformation and disinformation: Combating the spread of false information.
# User: Those are definitely important issues. Do you think the internet will continue to grow and evolve in the future?

# Llama 3: I believe the internet will continue to play a significant role in our lives. New technologies like artificial intelligence and virtual reality are likely to shape its future development.

# User: Thanks for the informative conversation, Llama! I learned a lot.
    
# Llama 3: You're welcome! I'm happy to help in any way I can

# demo.add_user_message("Hi Llama, can you tell me about the history of the internet?")

# demo.add_ai_message("Sure! The internet has a long and fascinating history. It began as a network of computers developed by the US Department of Defense in the 1960s. Over time, it grew and evolved, becoming the global network we know today.")

# demo.add_user_message("That's interesting! What were some of the key milestones in its development?")

# demo.add_ai_message("""
# Llama 3: Some key milestones include:

#  1969: The first ARPANET message is sent between two computers at UCLA and Stanford.
#  1983: TCP/IP protocol is adopted, which becomes the foundation of the modern internet.
#  1991: The World Wide Web is invented by Tim Berners-Lee.
#  2000s: The internet experiences rapid growth and becomes an essential part of daily life. """)

# demo.add_user_message("User: Thanks for the informative conversation, Llama! I learned a lot.")

# demo.add_ai_message("Llama 3: You're welcome! I'm happy to help in any way I can")

# demo.add_user_message("did Tim Berners-Lee have any help in developing it?")

# chain = prompt | llm

# response = chain.invoke(
#     {
#         "messages": demo.messages,
#     }
# )
# print(response)

# one for uploading

#another api for asking questions
