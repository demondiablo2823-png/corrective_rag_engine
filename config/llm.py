from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config.models import *
from dotenv import load_dotenv
load_dotenv()
import os
api=os.getenv("GROQ_API_KEY")

def get_llm():
    return ChatGroq(model=groq_model,api_key=api)

_output_parser = None
def output_parser():
    global _output_parser
    if _output_parser is None:
        _output_parser = StrOutputParser()
    return _output_parser