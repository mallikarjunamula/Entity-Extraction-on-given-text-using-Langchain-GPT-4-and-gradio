import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from typing import Optional
import dotenv
dotenv.load_dotenv()

# Pydantic data class
class Properties(BaseModel):
    TransactionMode: Optional[str]
    IFSCCode: str
    AccountNumber: int
    AccountName: str
    BankName: str
    OtherParty: Optional[str]

def entity_extraction(question):
    # Run chain
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = create_extraction_chain_pydantic(pydantic_schema=Properties, llm=llm)
    return chain.run(question)

entity = gr.Interface(
    entity_extraction,
    [
      gr.Textbox(label="Text/Narration:", value=""),
    ],
    "textbox",
    title="Enity Extraction from Text(Bank Narrations) using Langchain and OpenAI's GPT-4",
    theme = "gradio/monochrome"
)
entity.launch()

