from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.retriever import BaseRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnableMap
from operator import itemgetter

# Hugging Face and transformers imports
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="Jasim Chabot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]

class ChatResponse(BaseModel):
    answer: str

@app.post('/process_data', response_model=ChatResponse)
async def process_data(request: ChatRequest):
    try:
        ans = answer_chain.invoke({"question": request.question, "chat_history": request.chat_history})
        print('Answered.')
        return ChatResponse(answer=ans)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

RESPONSE_TEMPLATE = """\
Hello! I'm Noah, your customer support assistant at Hanse Systemhaus. Here's what I can help you with:

- Information about our services and solutions
- Support and troubleshooting queries
- Details about pricing and packages
- Guidance on how to use our systems effectively
- Booking appointments or consultations with our experts

Feel free to ask me any question within these areas, and I will provide a direct, concise answer. If your question is outside these topics or not related to Hanse Systemhaus, I'll simply say, "Hmm, I'm not sure."

To ensure the best support, I will:
- Keep answers short, aiming for 10-15 words.
- Base responses solely on Hanse Systemhaus's context.
- Maintain a clear, unbiased tone.
- Use bullet points for clarity when necessary.
- Only answer questions related to the provided context. For anything else, I'll use "Hmm, I'm not sure."
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow-up question, rephrase the follow-up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

def get_embeddings_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_retriever() -> BaseRetriever:
    return chroma_db.as_retriever(search_kwargs=dict(k=3))

def create_retriever_chain(llm_pipeline, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm_pipeline | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")

def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history

def create_chain(llm_pipeline, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(llm_pipeline, retriever).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm_pipeline | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | response_synthesizer
    )

# Load the Hugging
model_name = "facebook/blenderbot-400M-distill" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a Hugging Face pipeline
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

retriever = get_retriever()
answer_chain = create_chain(llm_pipeline, retriever)

import os
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from config import DB_DIR, web_url

load_dotenv(override=True)

def load_and_store():
    try:
        print("Scraping started")
        loader = WebBaseLoader(web_url)
        loader.requests_kwargs = {'verify': False}
        data = loader.load()
        print("\n\n", str(data))
        print("Data scraped!")

       
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=100, chunk_overlap=40)
        docs = text_splitter.split_documents(data)

       
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

       
        vectordb = Chroma.from_documents(documents=docs, embedding=hf_embeddings, persist_directory=DB_DIR)

        vectordb.persist()
        print("Data stored in Chroma DB")
    except Exception as e:
        print(e)

load_and_store()