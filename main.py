import json
import os
from langchain.agents import create_agent
from model.agent.tools import *
from langchain.agents import AgentState

import random
import numpy as np
import torch

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

tools = [call_api, ]

class MyState(AgentState):
    user_id: str
    call_count: int

def create_agent_versions(config_address: str, selected_agent: int)-> dict:
    with open(config_address, "r") as f:
        configs = f.readlines()
        dictionary = {c.split(":")[0].strip(): c.split(":")[1].strip() for c in configs}
        with open(dictionary["models"], "r") as f:
            agent_versions = json.load(f)
    
    return agent_versions["models"][selected_agent]


if __name__ == "__main__":
    config_address = "api.config"
    selected_agent: int = 0
    agent_version = create_agent_versions(config_address, selected_agent)

    agent = create_agent(
        model=agent_version, 
        tools=tools, 
        state_schema=MyState
    )
    set_seed(42)
