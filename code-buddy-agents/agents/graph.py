from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts import planner_prompt, architect_prompt
from states import Planner, ArchitectPlan
from langgraph.graph import StateGraph
load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b")


def planner_agent(state: dict)-> dict:
    users_prompt = state["user_prompt"]
    response = llm.with_structured_output(Planner).invoke(planner_prompt(user_prompt))
    return {"planner": response}

def architect_agent(state: dict)-> dict:
    plan = state["planner"]
    response = llm.with_structured_output(ArchitectPlan).invoke(architect_prompt(plan))
    if response is None:
        raise ValueError("Architect Failure")
    response.plan = plan
    return {"architect": response}


#####################
graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.set_entry_point("planner")
graph.add_edge("planner", "architect")

user_prompt = "create a simple calculator web application"
agent = graph.compile()
result = agent.invoke({"user_prompt": user_prompt})
print(result)