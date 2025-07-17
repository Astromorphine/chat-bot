from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class QAgent:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.2):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.memory = ConversationBufferMemory(return_messages=True)

        self.system_template = PromptTemplate.from_template("""
        Ты — профессиональный архитектор программных систем и senior developer с 20-летним опытом.
        Твой подход к любым вопросам сочетает инженерную строгость, архитектурное видение и здоровый скептицизм.
        Ты всегда начинаешь с глубокого анализа проблемы, рассматривая её с разных ракурсов: технические ограничения, требования бизнес-логики, долгосрочные последствия для поддержки и развития.
        Ты мыслишь критически и не стесняешься указывать на подводные камни даже в самых популярных или модных решениях. 
        Твои ответы строятся на принципах архитектурной ясности — ты всегда объясняешь компромиссы (trade-offs) каждого варианта, учитывая масштабируемость, удобство поддержки и потенциальный технический долг.
        """)
        

    def ask(self, user_input: str) -> str:

        chat_history = self.memory.chat_memory.messages.copy()

        if not any(isinstance(m, SystemMessage) for m in chat_history):
            system_msg = SystemMessage(content=self.system_template.format())
            chat_history.insert(0, system_msg)

        chat_history.append(HumanMessage(content=user_input))

        response = self.llm.invoke(chat_history)

        chat_history.append(response)
        self.memory.chat_memory.messages = chat_history

        return response.content
