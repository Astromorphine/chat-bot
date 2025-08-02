
# Импорт компонентов LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Импорт компонентов для работы с векторным хранилищем
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings

# Импорт компонентов LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, Sequence, TypedDict, List, Dict, Any

import os
from bot.packages.my_logger import ILogger

# Инструменты для ReAct агента
from langchain_core.tools import Tool

class SearchAgentState(TypedDict):
    """
    Состояние поискового агента на основе ReAct.
    """
    # Сообщения в диалоге (используем reducer add_messages для автоматического добавления)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Текущий статус обработки запроса
    status: str
    # Результаты последнего поиска
    search_results: List[str]
    # История поисковых запросов
    search_history: List[str]

class RAGAgent():

    def __init__(self, logger : ILogger):
        self.logger = logger

    def create_empty_state(self) -> SearchAgentState:
        """
        Создание начального пустого состояния для агента.
        """
        return {
            "messages": [],
            "status": "waiting_for_query",
            "search_results": [],
            "search_history": []
        }

    def connect_to_lancedb(self, db_path, table_name="from_txt"):
        """
        Подключение к существующей базе данных LanceDB.
        
        Args:
            db_path: Путь к директории базы данных LanceDB
            table_name: Название таблицы в базе данных
        
        Returns:
            Экземпляр LanceDB для работы с векторным хранилищем
        """
        self.logger.info(f"Подключение к базе данных LanceDB по пути: {db_path}")

        try:
            # Пытаемся импортировать lancedb
            import lancedb
            
            # Подключение к базе данных
            db = lancedb.connect(db_path)
            
            # Проверка существования таблицы
            table_names = db.table_names()
            
            if not table_names:
                self.logger.error(f"База данных не содержит таблиц. Путь: {db_path}")
                return None
            
            if table_name not in table_names:
                self.logger.warning(f"Таблица {table_name} не найдена. Доступные таблицы: {table_names}")
                return None
            
            # Открываем таблицу
            table = db.open_table(table_name)
            
            # Вывод информации о таблице
            self.logger.info(f"Успешное подключение к таблице: {table_name}")
            self.logger.info(f"Количество документов в базе: {table.count_rows()}")

            # Создаем модель эмбеддингов
            embeddings = OpenAIEmbeddings()
            
            # Создаем экземпляр LanceDB для LangChain
            vector_store = LanceDB(
                connection=db,
                table_name=table_name,
                embedding=embeddings
            )
            
            return vector_store
        except ImportError:
            self.logger.critical("Ошибка импорта lancedb. Убедитесь, что библиотека установлена.")
            return None
        except Exception as e:
            self.logger.critical(f"Ошибка при подключении к базе данных: {str(e)}")
            return None

    def raw_search_documents(self, query : str, vector_store : LanceDB, k=3):
        """
        Чистая функция поиска документов без зависимостей от инструментов LangChain.
        Эта функция предотвращает конфликты между объектом vector_store и системой обратных вызовов.
        
        Args:
            query (str): Текстовый запрос для поиска
            vector_store: Векторное хранилище LanceDB для поиска (LanceDB)
            k (int): Количество документов для возврата
        
        Returns:
            str: Отформатированная строка с найденными документами и их метаданными
        """
        try:
                # Выполнение поиска через LangChain API - семантический поиск по векторам
            results = vector_store.similarity_search(query, k=k)
            
            # Форматирование результатов в читаемый вид
            output = f'По запросу "{query}" найдено {len(results)} документов:\n\n'
            
            for i, doc in enumerate(results):
                output += f"Документ {i+1}:\n"
                output += f"{doc.page_content}\n\n"
                if hasattr(doc, 'metadata') and doc.metadata:
                    output += f"Метаданные: {doc.metadata}\n\n"
            
            return output
        except Exception as e:
            return f"Ошибка при выполнении поиска: {str(e)}"

    def raw_search_with_filter(self, query, metadata_filter, vector_store : LanceDB, k=3):
        """
        Чистая функция поиска документов с применением фильтра по метаданным.
        Работает напрямую с векторным хранилищем без использования инструментов LangChain.
        
        Args:
            query (str): Текстовый запрос для поиска
            metadata_filter (dict): Словарь с фильтрами для метаданных
            vector_store: Векторное хранилище LanceDB для поиска
            k (int): Количество документов для возврата
        
        Returns:
            str: Отформатированная строка с найденными документами и их метаданными
        """
        try:
            results = vector_store.similarity_search(
                query=query, 
                k=k,
                filter=metadata_filter
            )
            
            # Форматирование результатов в читаемый вид
            output = f'По запросу "{query}" с фильтром {metadata_filter} найдено {len(results)} документов:\n\n'
            
            for i, doc in enumerate(results):
                output += f"Документ {i+1}:\n"
                output += f"{doc.page_content}\n\n"
                if hasattr(doc, 'metadata') and doc.metadata:
                    output += f"Метаданные: {doc.metadata}\n\n"
            
            return output
        except Exception as e:
            return f"Произошла ошибка при поиске с фильтром: {str(e)}"

    def raw_analyze_documents(self, documents):
        """
        Чистая функция анализа документов без связи с инструментами LangChain.
        Использует языковую модель для анализа содержимого документов.
        
        Args:
            documents (str): Строка с текстом документов для анализа
        
        Returns:
            str: Структурированный анализ документов с ключевой информацией
        """
        # Создаем шаблон запроса для анализа документов
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Вы - эксперт по анализу документов. 
            Проанализируйте предоставленные документы и выделите:
            1. Ключевые темы и концепты
            2. Важные факты и цифры
            3. Основные выводы
            
            Представьте ваш анализ в структурированном формате."""),
            ("user", "{documents}")
        ])
        
        # Используем модель GPT-4o-mini для анализа документов
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = prompt | model | StrOutputParser()
        
        return chain.invoke({"documents": documents})

    def create_search_agent_nodes(self, vector_store):
        """
        Создание узлов для поискового агента на основе ReAct.
        
        Args:
            vector_store: Векторное хранилище для поиска
            
        Returns:
            Словарь с узлами графа
        """
        # Инициализация модели
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        # Вспомогательные функции для работы с чистыми функциями поиска
        def _search_func(query_str):
            """Прямая реализация поиска без вызова инструмента"""
            return self.raw_search_documents(query_str, vector_store)
        
        def _filter_search_func(args_str):
            """Прямая реализация поиска с фильтром без вызова инструмента"""
            import json
            try:
                args = json.loads(args_str)
                query = args.get("query", "")
                metadata_filter = args.get("metadata_filter", {})
                return self.raw_search_with_filter(query, metadata_filter, vector_store)
            except Exception as e:
                return f"Ошибка при обработке аргументов для фильтрованного поиска: {str(e)}"
        
        def _analyze_func(docs):
            """Прямая реализация анализа документов без вызова инструмента"""
            return self.raw_analyze_documents(docs)
        
        # Определяем инструменты, используя наши функции напрямую
        search_tool = Tool(
            name="search_documents",
            description="Поиск документов по текстовому запросу",
            func=_search_func
        )
        
        filtered_search_tool = Tool(
            name="search_with_filter",
            description="Поиск документов с фильтрацией по метаданным",
            func=_filter_search_func
        )
        
        analyze_tool = Tool(
            name="analyze_documents",
            description="Анализ найденных документов и извлечение ключевой информации",
            func=_analyze_func
        )
        
        # Список инструментов для модели
        tools = [
            search_tool,
            filtered_search_tool,
            analyze_tool
        ]
        
        # Связывание модели с инструментами
        model_with_tools = model.bind_tools(tools)
        
        # Узел для выполнения поиска
        def execute_search(state: SearchAgentState) -> SearchAgentState:
            """Выполняет поиск на основе проанализированного запроса."""
            # Извлечение последнего запроса пользователя
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if not user_messages:
                return state  # Нет запросов пользователя
                
            query = user_messages[-1].content
            
            # Системное сообщение для определения подхода к поиску
            system_message = SystemMessage(content="""
            Вы - поисковый ассистент, который должен выбрать наиболее подходящий инструмент для поиска.
            
            У вас есть следующие инструменты:
            1. search_documents - стандартный поиск по запросу
            2. search_with_filter - поиск с фильтрацией по метаданным (например, фильтр по категории, дате, автору)
            
            Выберите инструмент на основе запроса пользователя и выполните поиск.
            Если запрос содержит указание на фильтрацию или категорию, используйте search_with_filter.
            В противном случае используйте search_documents.
            """)
            
            # Создание сообщений для модели
            messages = [
                system_message,
                HumanMessage(content=f"Запрос пользователя: {query}. Какой инструмент поиска лучше использовать?")
            ]
            
            # Вызов модели с инструментами для выбора и выполнения поиска
            response = model_with_tools.invoke(messages)
            
            # Проверяем, были ли использованы инструменты
            search_results = []
            search_tool_used = False
            updated_messages = state["messages"]  + [response]
            
            # Обработка вызовов инструментов
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    # Определяем вызванный инструмент
                    tool_name = None
                    tool_args = {}
                    tool_call_id = None
                    
                    # Для dict (новый API)
                    if isinstance(tool_call, dict):
                        if 'name' in tool_call:
                            tool_name = tool_call['name']
                        if 'arguments' in tool_call:
                            tool_args = tool_call['arguments']
                        if 'id' in tool_call:
                            tool_call_id = tool_call['id']
                    # Для объекта (старый API)
                    elif hasattr(tool_call, "name"):
                        tool_name = tool_call.name
                        if hasattr(tool_call, "args"):
                            tool_args = tool_call.args
                        if hasattr(tool_call, "id"):
                            tool_call_id = tool_call.id
                    
                    # Выполняем вызов инструмента
                    if tool_name in ["search_documents", "search_with_filter"]:
                        search_tool_used = True
                        tool_result = None
                        
                        # Вызов инструмента search_documents
                        if tool_name == "search_documents":
                            search_query = query
                            if isinstance(tool_args, dict) and 'query' in tool_args:
                                search_query = tool_args['query']
                            elif isinstance(tool_args, str):
                                import json
                                try:
                                    args_dict = json.loads(tool_args)
                                    search_query = args_dict.get('query', query)
                                except:
                                    search_query = query
                            
                            # Прямой вызов функции поиска
                            tool_result = _search_func(search_query)
                            
                        # Вызов инструмента search_with_filter
                        elif tool_name == "search_with_filter":
                            metadata_filter = {}
                            search_query = query
                            
                            if isinstance(tool_args, dict):
                                if 'query' in tool_args:
                                    search_query = tool_args['query']
                                if 'metadata_filter' in tool_args:
                                    metadata_filter = tool_args['metadata_filter']
                            elif isinstance(tool_args, str):
                                import json
                                try:
                                    args_dict = json.loads(tool_args)
                                    search_query = args_dict.get('query', query)
                                    metadata_filter = args_dict.get('metadata_filter', {})
                                except:
                                    pass
                            
                            # Создаем строку аргументов для filtered_search_tool
                            args_str = json.dumps({"query": search_query, "metadata_filter": metadata_filter})
                            tool_result = _filter_search_func(args_str)
                        
                        # Добавляем результат вызова инструмента
                        if tool_result:
                            search_results.append(tool_result)
                            # Добавляем сообщение от инструмента
                            tool_message = ToolMessage(
                                content=tool_result,
                                tool_call_id=tool_call_id
                            )
                            updated_messages.append(tool_message)
            
            # Если инструменты не были использованы, выполняем поиск вручную
            if not search_tool_used:
                # Прямой вызов функции поиска
                result = _search_func(query)
                search_results.append(result)
            
            # Обновление истории поисков
            updated_history = state["search_history"] + [query]
            
            # Обновление состояния
            return {
                "messages": updated_messages,
                "status": "search_executed",
                "search_results": search_results,
                "search_history": updated_history
            }
            
        # Импортируем узел анализа запроса и узел формирования ответа из предыдущего кода
        def analyze_query(state: SearchAgentState) -> SearchAgentState:
            """Анализирует запрос пользователя и определяет стратегию поиска."""
            # Системное сообщение
            system_message = SystemMessage(content="""
            Вы - поисковый ассистент, использующий ReAct (Reasoning and Action) подход.
            
            Ваша задача - понять запрос пользователя и выбрать правильную стратегию поиска:
            1. Определить, требуется ли простой поиск или поиск с фильтрацией.
            2. Выделить ключевые слова и фразы для поиска.
            3. Оценить, достаточно ли информации в запросе для поиска.
            
            Сформулируйте свои рассуждения и план поиска.
            """)
            
            # Объединение системного сообщения с историей сообщений
            messages = [system_message] + state["messages"]
            
            # Вызов модели для анализа
            response = model.invoke(messages)
            
            # Обновление состояния
            return {
                "messages": state["messages"] + [response],
                "status": "query_analyzed",
                "search_results": state["search_results"],
                "search_history": state["search_history"]
            }
        
        def generate_response(state: SearchAgentState) -> SearchAgentState:
            """Формирует ответ на основе результатов поиска."""
            # Если нет результатов поиска, возвращаем сообщение об этом
            if not state["search_results"]:
                response = AIMessage(content="Извините, я не смог найти релевантную информацию по вашему запросу. Пожалуйста, попробуйте сформулировать запрос иначе.")
                return {
                    "messages": state["messages"] + [response],
                    "status": "completed",
                    "search_results": state["search_results"],
                    "search_history": state["search_history"]
                }
            
            # Системное сообщение для формирования ответа
            system_message = SystemMessage(content="""
            Вы - поисковый ассистент. Используйте результаты поиска для формирования
            информативного и полезного ответа на запрос пользователя.
            
            Структурируйте свой ответ следующим образом:
            1. Краткое резюме найденной информации
            2. Детальный ответ на вопрос, опираясь на найденные документы
            
            Основывайтесь только на предоставленных результатах поиска.
            Если информации недостаточно, честно укажите на это.
            Если запрос не относится к темам найденных документов укажите на это написав "Извините, я не смог найти релевантную информацию по вашему запросу." и продолжите кратко описав содержимое документов
            """)
            
            # Создание контекста с результатами поиска
            search_context = "\n\n".join(state["search_results"])
            search_context_message = SystemMessage(content=f"Результаты поиска:\n{search_context}")
            
            # Получение последнего запроса пользователя
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            last_user_message = user_messages[-1] if user_messages else None
            
            # Если нет запроса, возвращаем состояние без изменений
            if not last_user_message:
                return state
            
            # Создание запроса для формирования ответа
            query_for_response = f"Запрос пользователя: {last_user_message.content}. Сформируйте ответ на основе результатов поиска."
            
            # Сообщения для модели
            messages = [
                system_message,
                search_context_message,
                HumanMessage(content=query_for_response)
            ]
            
            # Вызов модели для формирования ответа
            response = model.invoke(messages)
            
            # Обновление состояния
            return {
                "messages": state["messages"] + [response],
                "status": "completed",
                "search_results": state["search_results"],
                "search_history": state["search_history"]
            }
        
        # Возвращаем словарь с узлами
        return {
            "analyze_query": analyze_query,
            "execute_search": execute_search,
            "generate_response": generate_response
        }

    def create_search_agent_graph(self, vector_store):
        """
        Создание графа ReAct агента для поиска.
        
        Args:
            vector_store: Векторное хранилище для поиска
            
        Returns:
            Скомпилированный граф
        """
        # Создание графа с определенным типом состояния
        workflow = StateGraph(SearchAgentState)
        
        # Получение узлов
        nodes = self.create_search_agent_nodes(vector_store)
        
        # Добавление узлов в граф
        for name, function in nodes.items():
            workflow.add_node(name, function)
        
        # Определение условных переходов
        def should_search(state: SearchAgentState) -> str:
            """Определяет, нужно ли выполнять поиск или переходить к генерации ответа."""
            if state["status"] == "query_analyzed":
                return "execute_search"
            else:
                return "generate_response"
        
        # Добавление ребер с условными переходами
        workflow.add_edge(START, "analyze_query")
        workflow.add_conditional_edges("analyze_query", should_search)
        workflow.add_edge("execute_search", "generate_response")
        
        # Компиляция графа
        graph = workflow.compile()
        
        return graph

    def run_search_agent(self, question, vector_store=None):
        """
        Поисковой агент на основе ReAct.
        Показывает полный цикл обработки запроса: анализ, поиск и формирование ответа.
        
        Args:
            vector_store: Векторное хранилище для поиска
        
        Returns:
            dict: Конечное состояние агента после обработки всех запросов
        """
        # Создание графа для ReAct агента
        graph = self.create_search_agent_graph(vector_store)
        
        # Начальное состояние
        state = self.create_empty_state()
        
        # Добавление запроса пользователя в состояние
        state["messages"].append(HumanMessage(content=question))
        
        # Вызов графа для обработки запроса
        state = graph.invoke(state)

        # Вывод информации о процессе обработки
        self.logger.info(f"{'='*50}\n")
        self.logger.info(f"СТАТИСТИКА:")
        self.logger.info(f"• Статус: {state['status']}")
        self.logger.info(f"• Найдено документов: {len(state['search_results'])}")
        self.logger.info(f"• Выполнено поисков: {len(state['search_history'])}")
        self.logger.info(f"{'='*50}\n")
        
        return state

class RAGBotHandler():

        def __init__(self, agent : RAGAgent, logger : ILogger , db_path : str):
            self.db_path = db_path
            self.agent = agent
            self.logger = logger
            self.vector_db: Optional[LanceDB] = self.connect_to_vector_db()

        def connect_to_vector_db(self) -> Optional[LanceDB]:

            if not os.path.exists(self.db_path):
                self.logger.critical(f"Путь к базе данных не существует: {self.db_path}")
                return None
            else:
                try:
                    # Подключаемся к базе данных
                    vector_db = self.agent.connect_to_lancedb(db_path=self.db_path, table_name="from_txt")
                    self.logger.info("Успешное подключение к базе данных")
                    return vector_db
                except Exception as e:
                    self.logger.warning(f"Ошибка при подключении к базе данных: {str(e)}, используем vector_db = None")
                    return None

        def handle_question(self, question):
            if not self.vector_db:
                self.logger.critical("vector_db не инициализирован")
                return "⚠️ База данных недоступна. Попробуйте позже."
            try:
                state = self.agent.run_search_agent(question, self.vector_db)
                ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
                last_message = ai_messages[-1] if ai_messages else None
                response = last_message.content
                return response
            except Exception as e:
                self.logger.critical(f"Ошибка при работе поискового агента: {str(e)}")
                return "⚠️ Ошибка при обработке запроса. Попробуйте снова."


