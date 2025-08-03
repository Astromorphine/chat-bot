import requests
from bs4 import BeautifulSoup
import os
import time
from pyhtml2pdf import converter
import re
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from common.paths import BASE_DIR, PDF_DIR, PDF_FILES

from playwright.async_api import async_playwright

class HTMLProcessing():
    def __init__(self):
        self.OUTPUT_DIRECTORY = BASE_DIR / "bank_data_output"
        self.HTML_DIRECTORY = BASE_DIR / "bank_data_output" / "html"
        self.PDF_DIRECTORY = PDF_DIR
        self.TEXT_DIRECTORY = BASE_DIR / "bank_data_output" / "text"
        self.doc_path = None

    async def generate_pdf(self, html_file, output_pdf):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(f'file:{html_file}')
            await page.pdf(path=output_pdf)
            await browser.close()

    # 1. Создаем все необходимые директории. Если они уже созданы то тоже все ок
    def create_output_folders(self, out_folder, html_folder, pdf_folder, text_folder):
        for directory in [self.OUTPUT_DIRECTORY, self.HTML_DIRECTORY, self.PDF_DIRECTORY, self.TEXT_DIRECTORY]:
            os.makedirs(directory, exist_ok=True)
        
        print("Директории для сохранения файлов созданы успешно")
        return

    # 2. Получение списка URL из карты сайта (sitemap.xml)
    def get_urls_from_sitemap(self, sitemap_url):
        """
        Получает список URL из sitemap.xml
        
        Args:
            sitemap_url (str): URL карты сайта
            
        Returns:
            list: Список URL из карты сайта
        """
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()
            # Используем lxml парсер специально для XML
            sitemap_soup = BeautifulSoup(response.content, 'lxml-xml')
            # Извлекаем все URL из тега <loc>
            urls = [loc.text for loc in sitemap_soup.find_all('loc')]
            print(f'Найдено {len(urls)} URL из карты сайта.')
            return urls
        except Exception as e:
            print(f'Ошибка при загрузке карты сайта: {e}')
            # Резервный список URL, если карта сайта недоступна
            backup_urls = [
                "https://www.tbank.ru/business/",
                "https://www.tbank.ru/business/help/",
                "https://www.tbank.ru/business/cards/"
            ]
            print(f'Используем резервный список из {len(backup_urls)} URL')
            return backup_urls

    # 3. Загрузка HTML-страниц и обработка кодировки
    def download_webpage(self, url):
        """
        Загружает HTML-страницу и корректно обрабатывает кодировку
        
        Args:
            url (str): URL страницы для загрузки
            
        Returns:
            tuple: (имя_файла, содержимое_html) или (имя_файла, None) при ошибке
        """
        print(f"Загрузка страницы: {url}")
        
        try:
            # Добавляем заголовки для имитации браузера и правильной обработки кодировки
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ru,en-US;q=0.7,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate, br'
            }
            response = requests.get(url, headers=headers, timeout=10)

            response.raise_for_status()  # Вызывает исключение при ошибке HTTP
            
            # Определяем кодировку страницы
            if response.encoding.upper() == 'ISO-8859-1':
                # Requests иногда неправильно определяет кодировку
                encoding = None
                # Проверяем кодировку в заголовке Content-Type
                content_type = response.headers.get('Content-Type', '')
                charset_match = re.search(r'charset=([\w-]+)', content_type)
                if charset_match:
                    encoding = charset_match.group(1)
                
                if not encoding:
                    # Ищем кодировку в meta-теге HTML
                    meta_match = re.search(r'<meta[^>]*charset=["\']*([^\"\'>]+)', response.text, re.IGNORECASE)
                    if meta_match:
                        encoding = meta_match.group(1)
                
                # Устанавливаем найденную кодировку или UTF-8 по умолчанию
                if encoding:
                    response.encoding = encoding
                else:
                    response.encoding = 'utf-8'
            
            # Генерируем имя файла из URL
            url_parts = urlparse(url)
            path = url_parts.path.rstrip('/')
            if path:
                filename = os.path.basename(path) or url_parts.netloc.replace('.', '_')
            else:
                filename = url_parts.netloc.replace('.', '_')
                
            # Добавляем .html, если нет расширения
            if '.' not in filename:
                filename += '.html'
            
            # Сохраняем HTML в файл
            filepath = os.path.join(self.HTML_DIRECTORY, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            print(f"HTML сохранен в {filepath}")
            self.doc_path = filepath
            return filename, response.text
        
        except Exception as e:
            print(f"Ошибка при загрузке {url}: {e}")
            if 'url_parts' in locals() and 'filename' in locals():
                return filename, None
            else:
                # Если произошла ошибка до создания имени файла, генерируем его из URL
                parsed_url = urlparse(url)
                path = parsed_url.path.rstrip('/')
                if path:
                    filename = os.path.basename(path) or parsed_url.netloc.replace('.', '_')
                else:
                    filename = parsed_url.netloc.replace('.', '_')
                    
                if '.' not in filename:
                    filename += '.html'
                    
                return filename, None

    # 4. Определение информативности страницы
    def is_content_page(self, html_content):
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
                print(f"✅ Информативная страница: {word_count} слов, {link_count} ссылок")
            else:
                print(f"❌ Не информативная страница: {word_count} слов, {link_count} ссылок")
                
            return is_informative
            
        except Exception as e:
            print(f"Ошибка при анализе страницы: {e}")
            return False

    # 5. Извлечение основного контента страницы с помощью BeautifulSoup
    def extract_main_content(self, html_content, url):
        """
        Извлекает основной контент страницы и форматирует его
        
        Args:
            html_content (str): HTML-содержимое страницы
            url (str): URL страницы
            
        Returns:
            str: Отформатированный HTML с основным контентом
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем ненужные элементы
            for tag in ['script', 'style', 'iframe', 'noscript']:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Ищем основной контент по популярным селекторам
            content_selectors = [
                'main', 'article', '.content', '#content', '.main-content', 
                '.page-content', '.container', '.article-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content and len(content.get_text(strip=True)) > 200:
                    main_content = content
                    print(f"Найден основной контент по селектору: {selector}")
                    break
                    
            # Если не нашли контент по селекторам, используем body
            if not main_content:
                main_content = soup.body
                print("Используем все содержимое тела страницы (body)")
            
            # Получаем заголовок страницы
            page_title = soup.title.string if soup.title else 'Банковская информация'
            
            # Создаем новый HTML-документ с форматированием
            formatted_html = f"""
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                <title>{page_title}</title>
                <style>
                    @charset "UTF-8";
                    body {{ 
                        font-family: 'Arial', sans-serif; 
                        line-height: 1.6; 
                        margin: 30px; 
                        color: #333; 
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1 {{ 
                        color: #0066cc; 
                        border-bottom: 1px solid #ddd; 
                        padding-bottom: 10px; 
                    }}
                    h2, h3, h4 {{ color: #0066cc; }}
                    a {{ color: #0066cc; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .footer {{ 
                        margin-top: 30px;
                        border-top: 1px solid #ddd;
                        padding-top: 10px;
                        font-size: 12px;
                        color: #666;
                    }}
                    @media print {{
                        body {{ font-size: 12pt; }}
                        a {{ text-decoration: none; color: #000; }}
                    }}
                </style>
            </head>
            <body>
                <h1>{page_title}</h1>
                {main_content}
                <div class="footer">
                    Источник: <a href="{url}">{url}</a><br>
                    Дата извлечения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </body>
            </html>
            """
            
            return formatted_html
            
        except Exception as e:
            print(f"Ошибка при извлечении контента: {e}")
            return html_content  # Возвращаем исходный HTML в случае ошибки

    # 6. Извлечение и форматирование текста
    def extract_text_content(self, html_content):
        """
        Извлекает и форматирует текст из HTML
        
        Args:
            html_content (str): HTML-содержимое страницы
            
        Returns:
            str: Извлеченный текст
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем ненужные элементы
            for tag in ['script', 'style', 'iframe', 'noscript']:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Получаем заголовок
            title = soup.title.string if soup.title else ''
            formatted_text = f"{title}\n{'='*len(title)}\n\n" if title else ""
            
            # Структурированное извлечение текста с сохранением иерархии
            # Обрабатываем заголовки и параграфы
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                tag_text = tag.get_text(strip=True)
                if tag_text:
                    if tag.name.startswith('h'):
                        # Заголовки выделяем в зависимости от уровня
                        level = int(tag.name[1])
                        formatted_text += f"\n{'#' * level} {tag_text}\n"
                    elif tag.name == 'p':
                        # Параграфы с двойным переносом строки
                        formatted_text += f"{tag_text}\n\n"
                    elif tag.name == 'li':
                        # Элементы списка с маркерами
                        formatted_text += f"- {tag_text}\n"
            
            # Если структурированное извлечение дало мало текста,
            # используем обычное извлечение текста
            if len(formatted_text) < 200:
                formatted_text = title + "\n\n" if title else ""
                formatted_text += soup.get_text(separator='\n', strip=True)
            
            # Очистка текста от лишних пробелов и переносов строк
            formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)  # Удаление лишних переносов
            formatted_text = re.sub(r'\s{2,}', ' ', formatted_text)     # Удаление лишних пробелов
            
            # Добавляем информацию об источнике в конец текста
            formatted_text += "\n\n----------\n"
            formatted_text += f"Дата извлечения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return formatted_text
            
        except Exception as e:
            print(f"Ошибка при извлечении текста: {e}")
            return ""
        
    # 7. Конвертация HTML в PDF с помощью WeasyPrint
    async def convert_html_to_pdf(self, html_content, output_file):
        """
        Конвертирует HTML в PDF с помощью WeasyPrint
        
        Args:
            html_content (str): HTML-содержимое для конвертации
            output_file (str): Путь для сохранения PDF
            
        Returns:
            bool: True в случае успеха, False в случае ошибки
        """
        try:
            print(f"Конвертация HTML в PDF: {output_file}") 
            
            # Сохраняем HTML во временный файл
            #temp_html_path = os.path.join(OUTPUT_DIRECTORY, 'temp_convert.html')
            temp_html_path = os.path.abspath('temp_convert.html')
            with open(temp_html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Конвертируем в PDF с помощью weasyprint
            await self.generate_pdf(temp_html_path, output_file)

            #converter.convert(f'file:{temp_html_path}', output_file)
            #html = weasyprint.HTML(filename=temp_html_path)
            #html.write_pdf(output_file)
            
            # Удаляем временный файл
            if os.path.exists(temp_html_path):
                os.remove(temp_html_path)
            
            # Проверяем, что PDF был создан
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"PDF успешно создан: {output_file}")
                return True
            else:
                print(f"Ошибка: PDF файл не создан или имеет нулевой размер")
                return False
            
        except Exception as e:
            print(f"Ошибка при конвертации в PDF: {e}")
            return False
        
    # 8. Основная функция обработки URL
    async def process_webpage(self, url):
        """
        Полная обработка одной веб-страницы: загрузка, извлечение контента, конвертация в PDF
        
        Args:
            url (str): URL страницы для обработки
            
        Returns:
            bool: True в случае успешной обработки, False в случае ошибки
        """
        print(f"\n=== Обработка URL: {url} ===\n")
        
        # Шаг 1: Загрузка HTML-страницы
        filename, html_content = self.download_webpage(url)
        
        if not html_content:
            print("❌ Не удалось загрузить HTML-страницу")
            return False
        
        # Шаг 2: Проверка информативности страницы
        if not self.is_content_page(html_content):
            print("⏩ Страница не содержит полезной информации, пропускаем")
            return False
        
        # Шаг 3: Извлечение основного контента
        formatted_html = self.extract_main_content(html_content, url)
        
        # Сохраняем очищенный HTML
        clean_html_filename = f"clean_{filename}"
        clean_html_path = os.path.join(self.HTML_DIRECTORY, clean_html_filename)
        with open(clean_html_path, 'w', encoding='utf-8') as f:
            f.write(formatted_html)
        print(f"✅ Очищенный HTML сохранен в {clean_html_path}")
        
        # Шаг 4: Извлечение текста
        extracted_text = self.extract_text_content(html_content)
        
        # Сохраняем извлеченный текст
        text_filename = os.path.splitext(filename)[0] + '.txt'
        text_path = os.path.join(self.TEXT_DIRECTORY, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"✅ Текст сохранен в {text_path}")
        
        # Шаг 5: Конвертация в PDF
        pdf_filename = os.path.splitext(filename)[0] + '.pdf'
        pdf_path = os.path.join(self.PDF_DIRECTORY, pdf_filename)
        success = await self.convert_html_to_pdf(formatted_html, pdf_path)

        if success:
            print(f"✅ Обработка URL завершена успешно")
            return pdf_filename
        else:
            print(f"❌ Обработка URL завершена с ошибками")
            return False

    # Сортировка URL по длине пути (количеству сегментов)
    def sort_urls_by_depth(self, urls):
        """Сортирует URLs по глубине (количеству сегментов)"""
        def get_path_depth(url):
            parsed = urlparse(url)
            # Считаем количество сегментов в пути
            segments = [s for s in parsed.path.split('/') if s]
            return len(segments)
        
        # Сортируем URL по убыванию глубины
        return sorted(urls, key=get_path_depth, reverse=True)

    # 9. Запуск обработки списка URL
    async def process_url(self, url, max_pages=10000):
        """
        Обрабатывает URL из карты сайта до достижения указанного количества
        информативных страниц
        
        Args:
            max_pages (int): Максимальное количество страниц для обработки
        """
        print(f"\n=== Начало обработки URL ===")

        print(f"\n{len(url)}] Проверка URL: {url}")
        
        # Проверяем, является ли страница информативной
        filename, html_content = self.download_webpage(url)
        
        if not html_content:
            print("❌ Не удалось загрузить страницу")
            return
            
        if self.is_content_page(html_content):

            print(f"🔍  Информативная страница : #{url}")
            
            # Обрабатываем найденную информативную страницу

            path = await self.process_webpage(url)

                
            # Если достигли лимита страниц, останавливаемся
        else:
            raise ValueError(f"Страница не является информативной: {url}")

        # Выводим итоговую статистику
        print("\nОбработка URL завершена==")

        self.doc_path = PDF_DIR / path


