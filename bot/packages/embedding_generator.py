from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import openai
import os
from typing import List

from bot.packages.i_classes.i_embedding_generator import IEmbeddingGenerator
from bot.packages.i_classes.i_embedding_generator import ILogger

class OpenAIEmbeddingGenerator(IEmbeddingGenerator):
    
    def __init__(self, logger : ILogger):

        self.logger = logger

        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не найден в .env")

        self.client = OpenAI(api_key=api_key)
        self.max_tokens = 8191  # пока не используется, но может пригодиться

    @retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APIConnectionError))
    )
    def create_embedding(self, text : str) -> list[float]:
        """
        Создает эмбеддинг для текста с использованием OpenAI API.
        
        Функция использует декоратор retry для автоматического повтора
        при ошибках API (rate limits, timeout и т.д.)
        
        Args:
            text: текст для создания эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        try:
            response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.critical(f"Произошла ошибка при создании вектора эмбеддинга, Trace:, {e}")
            raise

    def create_embeddings_for_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для списка чанков текста.
        
        Args:
            chunks: Список чанков текста.

        Returns:
            Список эмбеддингов для каждого чанка.
        """
        embeddings = []
        for chunk in chunks:
            embedding = self.create_embedding(chunk)
            embeddings.append(embedding)
        return embeddings
    