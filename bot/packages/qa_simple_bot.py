import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI

class QAgent():
    def __init__(self, model_name="gpt-4o-mini", temperature=0.2):
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.workflow = StateGraph(state_schema=MessagesState)
        self.init_agent()
        
    def call_model(self, state: MessagesState):
        response = self.model.invoke(state["messages"])
        # We return a list, because this will get added to the existing list
        return {"messages": response}
    
    def init_agent(self):
        # Define the two nodes we will cycle between
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self.call_model)

        # Adding memory is straight forward in langgraph!
        memory = MemorySaver()

        app = self.workflow.compile(
            checkpointer=memory
        )

        # The thread id is a unique key that identifies
        # this particular conversation.
        # We'll just generate a random uuid here.
        # This enables a single application to manage conversations among multiple users.
        thread_id = uuid.uuid4()
        config = {"configurable": {"thread_id": thread_id}}

        self.app = app
        self.config = config

    def ask(self, input_message : str):

        final_event = None
        input_message = HumanMessage(content=input_message)
        for event in self.app.stream({"messages": [input_message]}, self.config, stream_mode="values"):
            final_event = event

        return final_event["messages"][-1].content