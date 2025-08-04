import pandas as pd
import pyarrow as pa
import lancedb
from typing import List
from bot.packages.i_classes.i_logger import ILogger
from bot.packages.i_classes.i_vector_db import IVEctorDB
from bot.packages.i_classes.i_embedding_generator import IEmbeddingGenerator

from langchain_core.documents import Document
import hashlib

class LanceVectorDB(IVEctorDB):
    def __init__(self, logger: ILogger, embedding_generator : IEmbeddingGenerator) -> None:
        """
        Инициализация базы данных с логгером.
        """
        self.logger = logger
        self.embedding_generator = embedding_generator
        self.connection : lancedb.db.DBConnection
        self.current_table : lancedb.db.Table

    def get_connection(self) -> lancedb.db.DBConnection:
        if self.connection is None:
            self.logger.critical("❌ Нет соединения с базой данных!")
            raise ConnectionError("Database connection is not established")
        return self.connection
    
    def get_table(self) -> lancedb.db.Table:
        if not isinstance(self.current_table, lancedb.db.Table) or self.current_table == None:
            self.logger.critical("❌ Таблица lance не выбрана(self.current_table = None)")
            raise
        return self.current_table
    
    def connect_db(self, db_path: str = "data/lancedb"):
        """
        Создает или подключается к базе данных LanceDB.
        
        Args:
            db_path: путь к базе данных.
        """
        try:
            self.connection = lancedb.connect(db_path)
            if not self.get_connection():
                '''
                Метод в случае None вызовет raise, поэтому пока ничего не делаем
                Можно будет добавить какие либо действия в будущем
                '''
                pass
            self.logger.info(f"✅ Успешное подключение к базе данных {db_path}")
        except Exception as e:
            self.logger.critical(f"❌ Ошибка при подключении к базе данных: {e}")
            raise

    def select_table(self, table_name : str):
        try:
            self.current_table = self.connection.open_table(table_name)
            self.logger.info(f"✅ Успешное подключение к таблице: {table_name}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при подключении к таблице: {e}")
            raise

    def check_and_create_table(self, tablename : str):
        if not self.table_exists(tablename):
            self.create_table(tablename)

    def table_exists(self, tablename : str) -> bool:
        if not tablename in self.connection.table_names():
            return False
        return True

    def create_table(self, table_name: str):
        """
        Создает таблицу в LanceDB.
        
        Args:
            table_name: имя таблицы.
        """
        try:
            schema = pa.schema([
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 1536)),
                pa.field("doc_name", pa.string()),
                pa.field("chunk_id", pa.string())
            ])
            self.current_table = self.connection.create_table(table_name, schema=schema)
            self.logger.info(f"Таблица '{table_name}' успешно создана.")
        except Exception as e:
            self.logger.error(f"Ошибка при создании таблицы {table_name}: {e}")
            raise

    def generate_chunk_id(self, text: str, filename: str) -> str:
        # Хешируем текст и имя файла для уникальности
        combined_string = f"{filename}_{text}"
        return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()

    def fill_table(self, filename : str, chunks: List[Document], current_table : lancedb.db.Table):
        """
        Заполняет таблицу векторными данными.
        
        Args:
            filename: название источника чанков
            chunks: чанки для добавления
            current_table: объект таблицы для добавления
        """

        self.logger.info(f"Начинаем создание эмбеддингов и добавление в таблицу ({len(chunks)} чанков)...")
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            chunk_preview = chunk_text[:30].replace("\n", " ") + "..."
            self.logger.info(f"    Обработка чанка {i+1}/{len(chunks)}: '{chunk_preview}'")
            try:
                (f"    Создание эмбеддинга...")
                vector = self.embedding_generator.create_embedding(chunk_text)
                chunk_id = self.generate_chunk_id(chunk_text, filename)
                
                chunk_data = {
                    "text": chunk_text,
                    "vector": vector,
                    "doc_name": filename,
                    "chunk_id": chunk_id
                }
                self.logger.info(f"    Добавление в таблицу...")
                current_table.add([chunk_data])
                self.logger.info(f"    Чанк {i+1} успешно обработан и добавлен в таблицу.")
            except Exception as e:
                self.logger.warning(f"Ошибка при обработке чанка {i+1}: {str(e)}")
                self.logger.warning(f"   Тип ошибки: {type(e).__name__}")

    def display_search_results(self, results: pd.DataFrame):
        """
        Отображает результаты поиска в красивом формате.
        
        Args:
            results: DataFrame с результатами поиска.
        """
        if results.empty:
            self.logger.info("По вашему запросу ничего не найдено.")
            return

        i = 0
        self.logger.info(f"Найдено {len(results)} результатов:")
        for index, row in results.iterrows():
            text_preview = row['text'][:300].replace("\n", "<br>")
            self.logger.info(f"Результат #{i+1}:")
            self.logger.info(f"   Источник: {row['doc_name']}")
            self.logger.info(f"   ID чанка: {row['chunk_id']}")
            self.logger.info(f"   Релевантность: {row['_distance']:.4f}")
            self.logger.info(f"   Текст: {text_preview}...")
            i+=1

    def search_in_table(self, query_text, limit=3):
        """
        Поиск в таблице LanceDB по текстовому запросу.
        
        Этот метод автоматически:
        1. Создает эмбеддинг для запроса 
        2. Выполняет векторный поиск ближайших соседей
        3. Возвращает наиболее релевантные результаты
        
        Args:
            query_text (str): Текстовый запрос
            table: Таблица LanceDB для поиска
            limit (int): Количество результатов
            
        Returns:
            pandas.DataFrame: Результаты поиска
        """
        self.logger.info(f"Обрабатываем запрос: '{query_text}'")
        
        try:
            # Создаем эмбеддинг для запроса
            self.logger.info("Создаем эмбеддинг для запроса...")
            query_embedding = self.embedding_generator.create_embedding(query_text)
            self.logger.info(f"Эмбеддинг создан, размерность: {len(query_embedding)}")
            
            # Выполняем векторный поиск по эмбеддингу
            self.logger.info(f"Ищем {limit} наиболее релевантных чанков...")
            results = self.current_table.search(query_embedding).limit(limit).to_pandas()
            
            self.logger.info(f"Найдено {len(results)} результатов")
            return results

        except Exception as e:
            self.logger.warning(f"Ошибка при поиске, Trace: {e}")
            raise
