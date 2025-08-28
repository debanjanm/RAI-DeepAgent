import os

from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = ""


def load_language_model():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")
    return llm


def load_embedding_model():
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return emb
