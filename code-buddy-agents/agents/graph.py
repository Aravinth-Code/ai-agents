from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts import *
from states import *
from langgraph.constants import END
from langgraph.graph import StateGraph

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")


def planner_agent(state: dict)-> dict:
    users_prompt = state["user_prompt"]
    response = llm.with_structured_output(Planner).invoke(planner_prompt(user_prompt))
    return {"planner": response}


#####################
graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.set_entry_point("planner")

user_prompt = "create a simple calculator web application"
agent = graph.compile()
result = agent.invoke({"user_prompt": user_prompt})
print(result)