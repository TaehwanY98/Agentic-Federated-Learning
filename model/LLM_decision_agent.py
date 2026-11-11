from agent.tools import *
from langchain.agents import AgentState
from pydantic import BaseModel

tools = [call_api, ]

class AnswerDecision(BaseModel):
    heterogeneity : float
    evaluation_score : float
    description : str

class MyState(AgentState):
    user_id: str
    call_count: int

def AFL_Decision_Agent_Call(agent, num_samples: str, num_class:str, eval_metrics_scores:str,):
    result = agent.invoke({"messages": [{"role": "user", "content": "Calculate heterogeneity and task evaluation score in acting federated learning scenario with next message",
                                         "role": "user", "content": f"Number of samples : {num_samples}, Class Distribution : {num_class}, Evaluation Scores : {eval_metrics_scores}"
                                         }]})
    return result
