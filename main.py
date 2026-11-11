import os
from langchain.agents import create_agent
from model.agent.langchain import *
from langchain.agents import AgentState
from pydantic import BaseModel

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
tools = [call_api, ]

class AnswerDecision(BaseModel):
    heterogeneity : float
    evaluation_score : float
    description : str

class MyState(AgentState):
    user_id: str
    call_count: int

    
if __name__ == "__main__":
    agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools, state_schema=MyState)
    set_seed(42)
