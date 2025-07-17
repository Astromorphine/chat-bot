import os
import time
import random
import pandas as pd
import pyarrow as pa
import lancedb
from typing import List
from packages.i_classes.i_logger import ILogger
from packages.i_classes.i_vector_db import IVEctorDB
from packages.i_classes.i_embedding_generator import IEmbeddingGenerator
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown, HTML
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI
import openai

class LanceVectorDB(IVEctorDB):
    def __init__(self, logger: ILogger, embedding_generator : IEmbeddingGenerator, table_name: None ) -> None:
        """
        Инициализация базы данных с логгером.
        """
        self.logger = logger
        self.embedding_generator = embedding_generator
        self.db = None
        self.table = None
        self.table_name = table_name
        
        path = os.path.join(os.path.dirname(__file__), "..\\.env")
        load_dotenv(path)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


    def connect_db(self, db_path: str = "data/lancedb"):
        """
        Создает или подключается к базе данных LanceDB.
        
        Args:
            db_path: путь к базе данных.
        """
        try:
            self.db = lancedb.connect(db_path)
            self.logger.info(f"✅ Успешное подключение к базе данных {db_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при подключении к базе данных: {e}")
            raise

    def select_table(self):
        try:
            self.table = self.db.open_table(self.table_name)
            self.logger.info(f"✅ Успешное подключение к таблице: {self.table_name}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при подключении к nf,kbwt: {e}")
            raise

    def create_table(self, table_name: str = "pdf_chunks"):
        """
        Создает таблицу в LanceDB.
        
        Args:
            db_connection: соединение с базой данных
            table_name: имя таблицы.
        """
        try:
            # Определение схемы таблицы (с использованием PyArrow или схожего подхода)
            schema = pa.schema([
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 1536)),
                pa.field("doc_name", pa.string()),
                pa.field("chunk_id", pa.int32())
            ])
            self.table = self.db.create_table(table_name, schema=schema)
            self.logger.info(f"✅ Таблица '{table_name}' успешно создана.")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при создании таблицы {table_name}: {e}")
            raise

    def fill_table(self,table_name , chunks: List[dict]):
        """
        Заполняет таблицу векторными данными.
        
        Args:
            chunks: список чанков для добавления.
        """
        self.table = self.db.open_table(self.table_name)
        '''
        try:
            self.logger.info(f"🔄 Заполнение таблицы {table_name} чанками...")
            for i, chunk in enumerate(chunks):
                vector = self.embedding_generator.create_embedding(chunk.text)  # Генерация эмбеддинга для текста
                chunk = vector
                
                # Добавляем чанк в таблицу
                self.table.add([chunk])
                self.logger.info(f"✅ Чанк {i+1} добавлен.")
                time.sleep(random.uniform(0.5, 1.5))  # Небольшая задержка для избежания ограничения
        except Exception as e:
            self.logger.error(f"❌ Ошибка при добавлении чанков в таблицу: {e}")
            raise
        '''
        self.logger.info("🔄 Подготавливаем чанки для добавления...")
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            # Безопасное получение текста из чанка
            if hasattr(chunk, "text"):
                chunk_text = chunk.text
            else:
                chunk_text = str(chunk)
            
            # Получаем имя документа из метаданных или генерируем
            doc_name = "unknown"
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict) and "file_name" in chunk.metadata:
                doc_name = chunk.metadata["file_name"]
            else:
                doc_name = f"doc_{i // 2}"  # Простая группировка если нет метаданных
            
            processed_chunks.append({
                "text": chunk_text,
                "doc_name": doc_name,
                "chunk_id": i
            })

        progress = widgets.IntProgress(
            value=0,
            min=0,
            max=len(processed_chunks),
            description='Прогресс:',
            bar_style='info',
            orientation='horizontal'
        )
        display(progress)
        
        # Счетчик успешно добавленных чанков
        successful_chunks = 0
        print(processed_chunks)

        print(f"🚀 Начинаем создание эмбеддингов и добавление в таблицу ({len(processed_chunks)} чанков)...")
        for i, chunk in enumerate(processed_chunks):
            chunk_preview = chunk["text"][:30].replace("\n", " ") + "..."
            print(f"Обработка чанка {i+1}/{len(processed_chunks)}: '{chunk_preview}'")
            
            try:
                # Создаем эмбеддинг для текущего чанка
                print(f"   🧮 Создание эмбеддинга...")
                vector = self.create_embedding(chunk["text"])
                
                # Добавляем вектор к данным чанка
                chunk_to_add = chunk.copy()
                chunk_to_add["vector"] = vector
                
                # Добавляем чанк в таблицу
                print(f"   💾 Добавление в таблицу...")
                self.table.add([chunk_to_add])
                
                print(f"✅ Чанк {i+1} успешно обработан и добавлен в таблицу.")
                successful_chunks += 1
                
                # Обновляем прогресс
                progress.value = i + 1
                
                # Небольшая задержка между запросами для избежания rate limits
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                print(f"❌ Ошибка при обработке чанка {i+1}: {str(e)}")
                print(f"   Тип ошибки: {type(e).__name__}")
            
        print(f"\n🎉 Готово! {successful_chunks} из {len(processed_chunks)} чанков успешно добавлены в таблицу {table_name}")
        

    def search_in_table(self, query_text: str, limit: int = 3):
        """
        Выполняет поиск в таблице по текстовому запросу.

        Args:
            query_text: текст запроса.
            limit: максимальное количество результатов.
        """
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self.embedding_generator.create_embedding(query_text)
            self.logger.info(f"🧠 Эмбеддинг для запроса создан, размерность: {len(query_embedding)}")

            # Выполнение поиска
            results = self.table.search(query_embedding).limit(limit).to_pandas()
            self.logger.info(f"🔎 Найдено {len(results)} результатов.")

            return results
        except Exception as e:
            self.logger.error(f"❌ Ошибка при поиске: {e}")
            raise

    def display_search_results(self, results: pd.DataFrame):
        """
        Отображает результаты поиска в красивом формате.
        
        Args:
            results: DataFrame с результатами поиска.
        """
        if results.empty:
            self.logger.info("❌ По вашему запросу ничего не найдено.")
            return

        self.logger.info(f"🔍 Найдено {len(results)} результатов:")
        for i, row in results.iterrows():
            text_preview = row['text'][:300].replace("\n", "<br>")
            self.logger.info(f"Результат #{i+1}:")
            self.logger.info(f"   Источник: {row['doc_name']}")
            self.logger.info(f"   ID чанка: {row['chunk_id']}")
            self.logger.info(f"   Релевантность: {row['_distance']:.4f}")
            self.logger.info(f"   Текст: {text_preview}...")

    def interactive_search(self, db_path: str = "data/lancedb"):
        """
        Запускает интерактивный поиск в базе данных.
        
        Args:
            db_path: путь к базе данных.
            table_name: имя таблицы.
        """
        self.create_or_connect_db(db_path)
        self.create_table(self.db, self.table_name)

        while True:
            query_text = input("Введите запрос для поиска (или 'exit' для выхода): ")
            if query_text.lower() == 'exit':
                break

            results = self.search_in_table(query_text)
            self.display_search_results(results)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APIConnectionError))
    )
    def create_embedding(self, text):
        """
        Создает эмбеддинг для текста с использованием OpenAI API.
        
        Функция использует декоратор retry для автоматического повтора
        при ошибках API (rate limits, timeout и т.д.)
        
        Args:
            text: текст для создания эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """

        client = OpenAI(api_key=self.OPENAI_API_KEY)

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding
    

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
        print(f"🔍 Обрабатываем запрос: '{query_text}'")
        
        try:
            # Создаем эмбеддинг для запроса
            print("🧠 Создаем эмбеддинг для запроса...")
            query_embedding = self.create_embedding(query_text)
            print(f"✅ Эмбеддинг создан, размерность: {len(query_embedding)}")
            
            # Выполняем векторный поиск по эмбеддингу
            print(f"🔎 Ищем {limit} наиболее релевантных чанков...")
            results = self.table.search(query_embedding).limit(limit).to_pandas()
            
            print(f"📊 Найдено {len(results)} результатов")
            return results
            
        except Exception as e:
            print(f"❌ Ошибка при поиске: {str(e)}")
            raise