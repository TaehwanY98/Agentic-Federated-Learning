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
import json

class AgentCreatVersion(BaseTool):
    name : str = "create_agent_versions"
    description : str = "Generate [0, 6] integer value to selected_agent. Seed: 2024, Same user : Same value."
    config_address : str = "~/AgenticFL/Agentic-Federated-Learning/api.config"
    selected_agent : int


@tool(args_schema=AgentCreatVersion)
def create_agent_versions(config_address: str, selected_agent: int)-> dict:
    with open(config_address, "r") as f:
        configs = f.readlines()
        dictionary = {c.split(":")[0].strip(): c.split(":")[1].strip() for c in configs}
        with open(dictionary["models"], "r") as f:
            agent_versions = json.load(f)
    
    return agent_versions["models"][selected_agent]

class AgentCallAPI(BaseTool):
    name : str = "call_api"
    description : str = "Use a create_agent_versions tool, the output is a selected model. While using the model, you perform call_api tool with generated proper parameters by you"
    parameters : str = f"args : "
    model : str    

@tool(args_schema=AgentCallAPI)
def call_api(name: str , description:str, model: str, parameters: str):
    llm = init_chat_model(model)
    output = llm.invoke(description, config=parameters, )
    return output


if __name__ == "__main__":
    tools = [call_api,]
    agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools)