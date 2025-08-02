
import pytest
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

from bot.packages.html_processing import HTMLCleaner, HTMLDownloader
from bot.packages.my_logger import StandardLogger

@pytest.fixture
def mock_logger():
    return MagicMock(StandardLogger)

@pytest.fixture
def mock_downloader(mock_logger):
    downloader = HTMLDownloader(logger=mock_logger)
    downloader.download_with_playwright = MagicMock(return_value="Mocked Playwright HTML")
    downloader.download_with_requests = MagicMock(return_value="Mocked Requests HTML")
    return downloader

@pytest.mark.asyncio
async def test_download_with_playwright(mock_downloader, mock_logger):
    mock_downloader.requires_js_rendering = MagicMock(return_value=True)
    mock_downloader.download_with_playwright = AsyncMock(return_value = "Mocked Playwright HTML")
    mock_downloader.is_content_page = MagicMock(return_value=True)

    response, error = await mock_downloader.download("https://reactjs.org")

    assert response == "Mocked Playwright HTML"
    assert error is None

    mock_downloader.download_with_playwright.assert_called_once_with("https://reactjs.org")
    mock_logger.critical.assert_not_called()

@pytest.mark.asyncio
async def test_download_with_requests(mock_downloader, mock_logger):
    mock_downloader.requires_js_rendering = MagicMock(return_value=False)
    mock_downloader.is_content_page = MagicMock(return_value=True)
    mock_downloader.download_with_requests = MagicMock(return_value="Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, ")
    response, error = await mock_downloader.download("https://example.com")

    assert response == "Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, Mocked Requests HTML, "
    assert error is None
    
    mock_downloader.download_with_requests.assert_called_once_with("https://example.com")
    mock_logger.critical.assert_not_called()

@pytest.mark.asyncio
async def test_non_informative_page(mock_downloader, mock_logger):
    mock_downloader.requires_js_rendering = MagicMock(return_value=False)
    mock_downloader.is_content_page = MagicMock(return_value=False)  # Скажем, что это не контентная страница
    
    response, error = await mock_downloader.download("https://example.com")
    
    assert response is None
    assert error == "Ссылка не информативна"
    
    mock_logger.warning.assert_called_once_with("Загружена не информативная ссылка: https://example.com")

# Тестирование обработки ошибок при скачивании
@pytest.mark.asyncio
async def test_download_error(mock_downloader, mock_logger):
    mock_downloader.requires_js_rendering = MagicMock(return_value=False)
    mock_downloader.download_with_requests = MagicMock(side_effect=Exception("HTTP error"))
    
    response, error = await mock_downloader.download("https://example.com")
    
    # Проверяем, что возвращена ошибка
    assert response is None
    assert error == "Произошла ошибка при скачивании html"
    
    # Проверяем, что логируется критическая ошибка
    mock_logger.critical.assert_called_once_with("Произошла ошибка при скачивании html: HTTP error")

@pytest.mark.asyncio
async def test_empty_url(mock_downloader, mock_logger):
    mock_downloader.requires_js_rendering = MagicMock(return_value=False)
    response, error = await mock_downloader.download("")

    assert response is None
    assert error == "Указана неверная или пустая ссылка"