
# Необходимые библиотеки для работы с API telegram
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram import ReplyKeyboardMarkup
from telegram.constants import ChatAction

# Базовые библиотеки
import os
from dotenv import load_dotenv

# LangChain

from langchain_core.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from packages.app_context import AppContext
from packages.qa_simple_bot import QAgent

UPLOAD_FOLDER = "temp_files"
MODE_QUESTION = 0
MODE_RAG_QUESTION = 1
MODE_UPLOAD = 2
MODE_LINK_PARSE = 3
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_main_keyboard():
    return ReplyKeyboardMarkup(
        [["❓ Задать вопрос","📝 Поиск по базе знаний", "📤 Загрузить файл", "🌐 Парсинг страницы"]],
        resize_keyboard=True,
        one_time_keyboard=False
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выберите режим:", reply_markup=get_main_keyboard())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    text = update.message.text
    
    if text == "📤 Загрузить файл":
        await set_upload_mode(update, context)
        return
    elif text == "❓ Задать вопрос":
        await set_question_mode(update, context)
        return
    elif text == "🌐 Парсинг страницы":
        await set_link_parse_mode(update, context)
        return
    elif text == "📝 Поиск по базе знаний":
        await set_rag_search_mode(update, context)
        return

    current_mode = context.user_data.get("mode", MODE_QUESTION)

    if current_mode == MODE_QUESTION:

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        thinking_msg = await update.message.reply_text("🤔 Думаю...")

        agent = QAgent()
        response = agent.ask(text)

        await context.bot.delete_message(
            chat_id=update.effective_chat.id,
            message_id=thinking_msg.message_id
        )

        await update.message.reply_text(response, parse_mode="Markdown")

    elif current_mode == MODE_RAG_QUESTION:
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        thinking_msg = await update.message.reply_text("🤔 Думаю...")

        # Реализация ответа через app_context
        app_context = context.bot_data["app_context"]
        response = app_context.bot_handler.handle_question(text)

        await context.bot.delete_message(
            chat_id=update.effective_chat.id,
            message_id=thinking_msg.message_id
        )

        await update.message.reply_text(response, parse_mode="Markdown")

    elif current_mode == MODE_LINK_PARSE:
        
        if is_url_reachable(text):
            await update.message.reply_text("Ссылка валидна", parse_mode="Markdown")
            app_context = context.bot_data["app_context"]
            #tokenizer = app_context.tokenizer
            doc_proccesor = app_context.doc_proccesor
            #embedding_generator = app_context.embedding_generator
            html_poccessor = app_context.html_poccessor
            await html_poccessor.process_url(text)
            chunks = doc_proccesor.process_pdf_documents(html_poccessor.doc_path)
            app_context.lance_db.connect_db()
            app_context.lance_db.select_table()
            app_context.lance_db.fill_table("pdf_chunks",chunks)
        else:
            await update.message.reply_text("Ссылка не валидна", parse_mode="Markdown")

    else: 
        await update.message.reply_text(f"Измените режим на режим задачи вопросов")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if context.user_data.get("mode") != MODE_UPLOAD:
        await update.message.reply_text(
            "Сначала активируйте режим загрузки!",
            reply_markup=get_main_keyboard()
        )
        return

    file = await update.message.document.get_file()

    if file.file_size > 10 * 1024 * 1024:  # 10 МБ
        await update.message.reply_text("❌ Файл слишком большой.")
        return

    if not any(file.file_path.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        await update.message.reply_text("❌ Поддерживаются только PDF/DOCX/TXT.")
        return

    file_path = os.path.join(UPLOAD_FOLDER, update.message.document.file_name)
    await file.download_to_drive(file_path)

    response = f"{update.message.document.file_name} успешно сохранен"

    await update.message.reply_text(response)

    os.remove(file_path)

async def set_upload_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = MODE_UPLOAD
    await update.message.reply_text(
        "📤 Режим загрузки. Отправьте PDF/DOCX/TXT файл или нажмите «Задать вопрос».",
        reply_markup=get_main_keyboard()
    )

async def set_link_parse_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = MODE_LINK_PARSE
    await update.message.reply_text(
        "🌐 Режим парсинга страницы. Отправьте ссылку на ресурс и бот сможет сохранить содержимое страницы в базу знаний.",
        reply_markup=get_main_keyboard()
    )

async def set_question_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = MODE_QUESTION
    await update.message.reply_text(
        "❓ Режим вопросов. Напишите ваш запрос или нажмите «Загрузить файл».",
        reply_markup=get_main_keyboard()
    )

async def set_rag_search_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = MODE_RAG_QUESTION
    await update.message.reply_text(
        "📝 Поиск по базе знаний. Напишите ваш запрос, агент произведёт поиск по запросу.",
        reply_markup=get_main_keyboard()
    )

import requests

def is_url_reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

def main():

    path = os.path.join(os.path.dirname(__file__), "..\\.env")
    load_dotenv(path)
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app_context = AppContext()
    app.bot_data["app_context"] = app_context
    
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app_context.logger.info("Бот готов принимать запросы")

    app.run_polling()

if __name__ == "__main__":
    main()