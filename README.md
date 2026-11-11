# Agentic Federated Learning
- English
```text
This paper proposes "Agentic Federated Learning (AFL)," a novel approach that integrates LLM-driven decision-making with tool calling to address the complex challenges inherent in standard Federated Learning (FL). Traditionally, FL faces significant obstacles such as non-IID data distributions, high communication overheads, and non-adaptive aggregation methods, which can compromise model performance and efficiency. Our AFL framework introduces a "Functional Tools" suite within the server, encompassing vital components like an Aggregation Engine, Client Selector, and Parameter Tuner. Crucially, we leverage an LLM-based agent with a "think-and-act" loop to dynamically orchestrate these tools via tool calls, responding in real-time to evolving system states and client metrics. A sophisticated "Prompt Engineering" layer translates system objectives and monitor data into optimal tool selections and configurations. Our comprehensive framework incorporates features such as goal-based optimization (e.g., maximizing accuracy, minimizing latency), a comprehensive knowledge base, and client-side support for tool-call execution, all working in concert to create a highly flexible and powerful AFL system.
```

- Korean
```text
본 논문은 표준 연합 학습(Federated Learning, FL)의 복잡한 과제를 해결하기 위해 LLM 기반 의사 결정과 도구 호출(tool calling)을 통합한 혁신적인 접근 방식인 "에이전틱 연합 학습(Agentic Federated Learning, AFL)"을 제안한다. 기존의 연합 학습은 비독립적(non-IID) 데이터 분포, 높은 통신 오버헤드, 비적응형 집계 방식 등 상당한 장애물에 직면하여 모델 성능과 효율성을 저하시킬 수 있었다. 본 논문에서 제안하는 AFL 프레임워크는 집계 엔진, 클라이언트 선택기, 매개변수 조정기와 같은 필수 구성 요소를 포함하는 "기능 도구(Functional Tools)" 세트를 서버 내에 도입한다. 특히, 우리는 실시간으로 진화하는 시스템 상태 및 클라이언트 지표에 대응하기 위해 도구 호출을 통해 이러한 도구를 동적으로 조율하는 "사고 및 행동(think-and-act)" 루프를 갖춘 LLM 기반 에이전트를 활용한다. 정교한 "프롬프트 엔지니어링(Prompt Engineering)" 레이어는 시스템 목표와 모니터링 데이터를 최적의 도구 선택 및 구성으로 변환한다. 우리의 포괄적인 프레임워크는 목표 기반 최적화(예: 정확도 극대화, 지연 시간 최소화), 포괄적인 지식 기반, 도구 호출 실행을 위한 클라이언트 측 지원과 같은 기능을 통합하여 매우 유연하고 강력한 AFL 시스템을 구축한다.
```
![Proposed Algorithm Diagram](./res/diagram/Agentic%20FL%20Initial%20Diagram.png)


# How we build an AI agent?

### Documentation Index
[Document](https://docs.langchain.com/oss/python/deepagents/quickstart)
    
Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
Use this file to discover all available pages before exploring further.

# Agents

Build a LangChain agent: a model calling tools in a loop until a task is complete, shaped by a harness of prompt, tools, and middleware.

An agent is a model calling tools in a loop until a given task is complete.

![Agent loop diagram](https://mintcdn.com/langchain-5e9cc07a/jtty0O--UJOKG0nK/oss/images/core_agent_loop.svg?fit=max\&auto=format\&n=jtty0O--UJOKG0nK\&q=85\&s=4b4cbb497b6273758a565de1bc90ece0) 

A harness is everything around that loop: the prompt, the tools, and any middleware that shapes the model's behavior.

**Agent = Model + Harness**

The job of a harness: get the model the right context at the right time for the given task.

[`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) is a highly configurable harness. At its simplest, you can create one with:

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="openai:gpt-5.5", tools=tools)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools)
```

Building on that, you can configure the basics directly with the `model=`, `tools=`, and `system_prompt=` parameters. For more advanced capabilities, extend the harness with [middleware](#configure-the-harness).

[Managed Deep Agents](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview.md) builds on `create_agent` and comes with commonly useful capabilities already assembled, such as planning, file system tools, subagents, and memory. For maximum flexibility while using the same Deep Agents harness, use [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview.md) and host agents yourself, or run a custom [LangSmith Deployment](https://docs.langchain.com/langsmith/deployment.md). Use `create_agent` when you need to configure a smaller harness yourself.

## Core components

![Agent model and harness components diagram](https://mintcdn.com/langchain-5e9cc07a/jtty0O--UJOKG0nK/oss/images/agent_model_harness.svg?fit=max\&auto=format\&n=jtty0O--UJOKG0nK\&q=85\&s=5ac6a7e0343af7cb5ba3ca632e2224af)

### Model

Pass a model identifier string (`"provider:model"`) or an initialized model instance to select the model for your agent. See [Models](https://docs.langchain.com/oss/python/langchain/models.md) for parameters, provider setup, and dynamic model selection.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="openai:gpt-5.5", tools=tools)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools)
```

### Tools

To provide the agent with tools, pass any Python callable, LangChain tool, or tool dict. See [Tools](https://docs.langchain.com/oss/python/langchain/tools.md) for tool definition, context access, and dynamic tool selection.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="google_genai:gemini-3.6-flash", tools=[search])
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="openai:gpt-5.5", tools=[search])
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="anthropic:claude-sonnet-5", tools=[search])
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=[search])
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=[search])
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=[search])
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="ollama:north-mini-code-1.0", tools=[search])
```

### System prompt

Shape how the agent approaches tasks. The system prompt parameter accepts a string or `SystemMessage`. For dynamic prompts at runtime, use [middleware](https://docs.langchain.com/oss/python/langchain/middleware.md).

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model="openai:gpt-5.5",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

### Structured output

Return a validated schema from the agent using `response_format=`. See [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output.md) for strategies and examples.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="openai:gpt-5.5", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel
from langchain.agents import create_agent


class Answer(BaseModel):
    summary: str
    confidence: float


agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools, response_format=Answer)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
result["structured_response"]  # Answer(summary=..., confidence=...)
```

### Agent state

Every agent manages its execution context through [`AgentState`](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentState), a typed dictionary that holds the current conversation history and any custom fields your tools and middleware need.

The built-in field is:

| Field | Type | Description |
| - | - | - |
| `messages` | `list[BaseMessage]` | The full conversation history for the current thread. Append-only: new messages are added, never replaced. |

`AgentState` is also the type signature for every node-style middleware hook (`before_model`, `after_model`, and similar). Hooks receive the current state and can return a dict of updates to merge back into it.

To add custom fields (for example, a `user_id` or a counter), subclass `AgentState` and pass the subclass to `create_agent` via `state_schema=`:

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState, create_agent


class MyState(AgentState):
    user_id: str
    call_count: int


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[],
    state_schema=MyState,  # [!code highlight]
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState, create_agent


class MyState(AgentState):
    user_id: str
    call_count: int


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[],
    state_schema=MyState,  # [!code highlight]
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState, create_agent


class MyState(AgentState):
    user_id: str
    call_count: int


agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[],
    state_schema=MyState,  # [!code highlight]
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState, create_agent


class MyState(AgentState):
    user_id: str
    call_count: int


agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[],
    state_schema=MyState,  # [!code highlight]
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState, create_agent


class MyState(AgentState):
    user_id: str
    call_count: int


agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[],
    state_schema=MyState,  # [!code highlight]
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState, create_agent


class MyState(AgentState):
    user_id: str
    call_count: int


agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[],
    state_schema=MyState,  # [!code highlight]
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import AgentState, create_agent


class MyState(AgentState):
    user_id: str
    call_count: int


agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[],
    state_schema=MyState,  # [!code highlight]
)
```

For full details, examples, and middleware-level state schemas, see [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory.md#customizing-agent-memory) and [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom.md#state-updates).

## Invocation

Trace each step of this loop, debug tool calls, and evaluate agent outputs with [LangSmith](https://smith.langchain.com?utm_source=docs\&utm_medium=cta\&utm_campaign=langsmith-signup\&utm_content=oss-langchain-agents). Follow the [tracing quickstart](https://docs.langchain.com/langsmith/trace-with-langchain.md) to get set up. We recommend you also set up [LangSmith Engine](https://docs.langchain.com/langsmith/engine.md) which monitors your traces, detects issues, and proposes fixes.

You can invoke an agent with a message. Behind the scenes that passes an update to the agent's [`State`](https://docs.langchain.com/oss/python/langgraph/graph-api#state). All agents include a [sequence of messages](https://docs.langchain.com/oss/python/langgraph/graph-api-state.md#messagesstate) in their state; to invoke the agent, pass a new message along with a `thread_id` so the agent can persist and resume conversation history:

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

# A follow-up turn on the same conversation: reuse the same thread_id to keep history
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

[View example trace](https://smith.langchain.com/public/1835d2f3-8da5-4e27-95d7-fb2c7d726ad5/r)

Open a public LangSmith run for this example.
Persisting conversation history with `thread_id` requires the agent to be configured with a [checkpointer](https://docs.langchain.com/oss/python/langchain/long-term-memory.md). When deployed on [LangSmith](https://docs.langchain.com/langsmith/deployment.md), a checkpointer is provisioned automatically. Locally, pass one explicitly, for example `create_agent(..., checkpointer=InMemorySaver())`.

If you also need to pass per-run configuration (such as a user ID, API keys, or feature flags) to tools and middleware, pass it as `context` alongside `config`. Define the shape of that data with `context_schema` and access it through `runtime.context`:

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

[View example trace](https://smith.langchain.com/public/79519c6b-9286-45f0-b5c6-9afaa5ef22f8/r)

Open a public LangSmith run for this example.

`thread_id` scopes the *conversation* (message history, checkpoints), while `context` carries *per-run* data your tools and middleware read at invocation time. Both are commonly passed together. See [tool context](https://docs.langchain.com/oss/python/langchain/tools.md#context) and [Runtime](https://docs.langchain.com/oss/python/langchain/runtime.md) for more.

## Streaming

`invoke` returns the final response at the end of a run. If an agent executes multiple tool calls, users often need progress updates before completion. Use streaming to surface intermediate messages and tool activity as they happen.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.messages import AIMessage, HumanMessage


stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]},
    version="v3",
)
for snapshot in stream.values:
    # Each snapshot contains the full state at that point
    latest_message = snapshot["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

[View example trace](https://smith.langchain.com/public/25a56e3b-08d0-4fcd-934e-a16c0a99d629/r)

Open a public LangSmith run for this example.


For streaming modes, event types, and UI patterns, see [Streaming](https://docs.langchain.com/oss/python/langchain/streaming.md).

## Configure the harness

`create_agent` is highly extensible. Middleware is the primitive for customization: each piece handles one concern, hooks into the agent loop at the right moment, and composes freely with any other. Take exactly what your use case needs and skip the rest.

Common patterns are prebuilt as first-class middleware. You can build anything else as [custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom.md).

![Agent harness capabilities by category](https://mintcdn.com/langchain-5e9cc07a/jtty0O--UJOKG0nK/oss/images/agent_harness_capabilities.svg?fit=max\&auto=format\&n=jtty0O--UJOKG0nK\&q=85\&s=0ff671d72badd0844826660dfcb04391)

As agents take on complex work, they need support across a few key areas. The middleware ecosystem provides:

[Execution environment](#execution-environment)

Tools, filesystem, sandboxes, and code execution

[Context management](#context-management)

Summarization, memory, skills, and prompt caching

[Planning and delegation](#planning-and-delegation)

Todo lists and subagents for parallel, isolated work

[Fault tolerance](#fault-tolerance)

Retries, fallbacks, and call limits

[Guardrails](#guardrails)

PII detection and content controls

[Steering](#steering)

Human-in-the-loop approval before high-impact actions


 `create_deep_agent` pre-assembles this stack for long-running coding and research tasks (filesystem, summarization, subagents, and prompt caching included by default). See [Deep Agents](https://docs.langchain.com/oss/python/deepagents/harness.md) for the full prebuilt harness.

### Execution environment

Agents are especially useful when they can take action rather than just generate text. The execution environment gives the agent a workspace: tools it can call, a filesystem for reading and writing files across turns, and code execution for running scripts or shell commands.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

See [`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware), [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes.md), [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters.md).

> [!NOTE]
>
> This example imports from the `deepagents` package. Install it with:
>
> ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
> pip install deepagents
> ```
>
> ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
> uv add deepagents
> ```

### Context management

Every model call has a fixed context window. As an agent runs, that window fills with accumulating history, tool results, and intermediate steps. Summarization compresses history before overflow hits; memory loads persistent instructions at startup so knowledge carries across sessions; skills surface domain knowledge on demand rather than loading everything upfront.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

backend = StateBackend()
model = "google_genai:gemini-3.6-flash"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

backend = StateBackend()
model = "openai:gpt-5.5"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

backend = StateBackend()
model = "anthropic:claude-sonnet-5"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

backend = StateBackend()
model = "openrouter:z-ai/glm-5.2"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

backend = StateBackend()
model = "fireworks:accounts/fireworks/models/glm-5p2"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

backend = StateBackend()
model = "baseten:zai-org/GLM-5.2"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware

backend = StateBackend()
model = "ollama:north-mini-code-1.0"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

See [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware), [`MemoryMiddleware`](https://reference.langchain.com/python/deepagents/middleware/memory/MemoryMiddleware), [Skills](https://docs.langchain.com/oss/python/langchain/multi-agent/skills.md), [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering.md).


This example imports from the `deepagents` package. Install it with:

```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
pip install deepagents
 ```

```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
uv add deepagents
```

### Planning and delegation

Complex tasks often exceed what one context window can handle. Delegation lets the main agent break work into pieces, hand them to subagents that each run in their own isolated context, and stay focused on coordination rather than execution. Work can run in parallel; the main agent's context stays clean.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


backend = StateBackend()

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


backend = StateBackend()

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


backend = StateBackend()

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


backend = StateBackend()

agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


backend = StateBackend()

agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


backend = StateBackend()

agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


backend = StateBackend()

agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

See [Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents.md).



This example imports from the `deepagents` package. Install it with:

```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
pip install deepagents
```

```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
uv add deepagents
```

### Name your agent

Optionally use an identifier for the agent. This is especially useful when embedding the agent as a subgraph in [multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent.md) systems.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(model="google_genai:gemini-3.6-flash", tools=tools, name="research_assistant")
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(model="openai:gpt-5.5", tools=tools, name="research_assistant")
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(model="anthropic:claude-sonnet-5", tools=tools, name="research_assistant")
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(model="openrouter:z-ai/glm-5.2", tools=tools, name="research_assistant")
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(model="fireworks:accounts/fireworks/models/glm-5p2", tools=tools, name="research_assistant")
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(model="baseten:zai-org/GLM-5.2", tools=tools, name="research_assistant")
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_agent(model="ollama:north-mini-code-1.0", tools=tools, name="research_assistant")
```

### Fault tolerance

Agents in production encounter failures that rarely appear in development: rate limits, model timeouts, transient API errors. Fault tolerance middleware handles these at the infrastructure level so your tools and business logic don't need try/catch around every call.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

See [`ModelRetryMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/model_retry/ModelRetryMiddleware), [`ToolRetryMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/tool_retry/ToolRetryMiddleware), [Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in.md).

### Guardrails

Some policies can't live in a prompt—they need to be enforced deterministically regardless of what the model does. Guardrails intercept data as it flows through the agent loop, applying compliance rules or content policies before tool results reach the model's context.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[search],
    middleware=[PIIMiddleware("email")],
)
```

See [`PIIMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/pii/PIIMiddleware), [Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in.md).

### Steering

Full autonomy isn't always appropriate. Steering lets you place humans at specific decision points—before destructive writes, expensive API calls, or anything requiring judgment—without restructuring your agent. The agent pauses and waits; a human approves, edits, or rejects; execution continues.

```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
)
```

```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[search],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
)
```

```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
)
```

```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="openrouter:z-ai/glm-5.2",
    tools=[search],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
)
```

```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="fireworks:accounts/fireworks/models/glm-5p2",
    tools=[search],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
)
```

```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="baseten:zai-org/GLM-5.2",
    tools=[search],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
)
```

```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


agent = create_agent(
    model="ollama:north-mini-code-1.0",
    tools=[search],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
)
```

See [`HumanInTheLoopMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/human_in_the_loop/HumanInTheLoopMiddleware), [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop.md).

### Middleware resources

[Middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview.md)

How the middleware stack works and when hooks fire

[Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in.md)

Full reference with configuration examples

[Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom.md)

Write your own hooks for business logic, PII scrubbing, and more

***

[Connect these docs](https://docs.langchain.com/use-these-docs.md) to your agent of choice via MCP for real-time answers.

[Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/agents.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).


______
# Tool Abstraction
#### What is the concept of "Tools" in AI agent?
    Tools extend what agents can do—letting them fetch real-time data, execute code, query external databases, and take actions in the world. Under the hood, tools are callable functions with well-defined inputs and outputs that get passed to a chat model. The model decides when to invoke a tool based on the conversation context, and what input arguments to provide.

### Create tools

- Basic tool definition

    The simplest way to create a tool is with the @tool decorator. By default, the function’s docstring becomes the tool’s description that helps the model understand when to use it:

```python    
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

- Custom tool name

    By default, the tool name comes from the function name. Override it when you need something more descriptive. Override the auto-generated tool description for clearer model guidance:

```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

- Advanced schema definition

    Define complex inputs with Pydantic models or JSON schemas:
```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```