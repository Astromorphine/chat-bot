
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from bot.packages.i_classes.i_logger import ILogger

path = os.path.join(os.path.dirname(__file__), "..\\.env")
load_dotenv(path)


class TravilyAgent():

    def __init__(self, logger: ILogger):
        self.logger = logger

    def invoke(self, user_input : str):
        llm = init_chat_model("gpt-4o-mini", model_provider="openai")

        tavily_search_tool = TavilySearch(
            max_results=5,
            topic="general",
        )

        agent = create_react_agent(llm, [tavily_search_tool])

        response = agent.invoke({"messages": user_input})

        return response["messages"][-1].content
