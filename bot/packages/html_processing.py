import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from typing import Optional

from common.paths import TXT_DIR
from bot.packages.i_classes.i_logger import ILogger

import base64
from pathlib import Path

class HTMLDownloader():

    def __init__(self, logger: ILogger):
        self.logger = logger

    async def download(self, url: str) -> tuple[Optional[str],Optional[str]]:
        """
        Скачивает HTML контент с указанного URL.

        :param url: URL для скачивания
        :return: tuple, контент или None в случае ошибки, текст ошибки или None при её отсутствии
        """
        if not url or not isinstance(url, str):
            return (None, "Указана неверная или пустая ссылка")
            
        try:
            if self.requires_js_rendering(url):
                response = await self.download_with_playwright(url)
            else:
                response = self.download_with_requests(url)

            if self.is_content_page(response):
                return (response, None)
            else: 
                self.logger.warning(f"Загружена не информативная ссылка: {url}")
                return (None, "Ссылка не информативна")
            
        except Exception as e:
            self.logger.critical(f"Произошла ошибка при скачивании html: {e}")
            return (None, "Произошла ошибка при скачивании html")

    def requires_js_rendering(self, url: str) -> bool:
        # Простейшая проверка по URL (нужно доработать под реальные кейсы)
        return "react" in url or "vue" in url

    async def download_with_playwright(self, url: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            content = await page.content()
            await browser.close()
        return content

    def download_with_requests(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def encode_filename_base64(self, url: str) -> str:
        return base64.urlsafe_b64encode(url.encode('utf-8')).decode('utf-8')

    def decode_filename_base64(self, encoded_url: str) -> str:
        return base64.urlsafe_b64decode(encoded_url.encode('utf-8')).decode('utf-8')

    def create_txt(self, text : str, url : str) -> Path | None:
        try:
            file_path = TXT_DIR / f"{self.encode_filename_base64(url)}.txt"
            with open(file=file_path,mode="w", encoding="utf-8") as f:
                data = f.write(text)
                self.logger.info(f"Файл с названием: {url} и размером в {data} символов был успешно сохранён, путь -> {file_path}")
            return file_path
        except Exception as e:
            self.logger.critical(f"Произошла ошибка при сохранении txt файла, Trace: {e}")
            return None

    def is_content_page(self, html_content) -> bool:
        """
        Определяет, является ли страница информативной (содержит полезный контент)
        
        Args:
            html_content (str): HTML-содержимое страницы
            
        Returns:
            bool: True, если страница содержит полезную информацию
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем скрипты и стили для точного подсчета текста
            for tag in ['script', 'style']:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Получаем текст страницы
            text = soup.get_text(separator=' ', strip=True)
            
            # Аналитика страницы
            word_count = len(text.split())
            link_count = len(soup.find_all('a', href=True))
            link_to_text_ratio = link_count / word_count if word_count > 0 else float('inf')
            
            # Критерии информативной страницы
            is_informative = (
                # Много текста
                word_count > 300 or
                # Среднее количество текста с малым количеством ссылок
                (word_count > 100 and link_count < 15) or
                # Хорошее соотношение текста к ссылкам
                (word_count > 150 and link_to_text_ratio < 0.1)
            )
            
            if is_informative:
                self.logger.info(f"✅ Информативная страница: {word_count} слов, {link_count} ссылок")
            else:
                self.logger.info(f"❌ Информативная страница: {word_count} слов, {link_count} ссылок")
                
            return is_informative
            
        except Exception as e:
            self.logger.critical(f"Ошибка при анализе страницы: {e}")
            return False

class HTMLCleaner():

    def __init__(self, logger: ILogger):
        self.logger = logger

    def clean(self, html: str) -> str:
        """
        Очищает HTML от ненужных тегов (стилей, скриптов, навигации) и извлекает полезный текст.

        :param html: HTML-контент
        :return: Очищенный текст
        """
        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup(["script", "style", "footer", "header", "nav", "aside", "form", "button"]):
            tag.decompose()  

        clean_text = soup.get_text(separator="\n", strip=True)

        clean_text = "\n".join(line.strip() for line in clean_text.split("\n") if line.strip())

        return clean_text




