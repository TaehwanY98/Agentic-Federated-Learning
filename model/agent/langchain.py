from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import (
    BaseTool,
    InjectedState,
    InjectedStore,
    InjectedToolArg,
    InjectedToolCallId,
    ToolException,
    ToolRuntime,
    tool
    )

class AgentCallAPI(BaseTool):
    name : str = "call_api"
    description : str = "Call an API with the given parameters."
    parameters : str = f"args : "
    model : str = "google_genai:gemini-3.6-flash"

@tool(args_schema=AgentCallAPI)
def call_api(query:str, model: str, parameters: str):
    llm = init_chat_model(model)
    output = llm.invoke(query, config=parameters)
    return output
