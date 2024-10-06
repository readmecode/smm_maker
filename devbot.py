import os
import logging
import asyncio
from collections import deque
from dotenv import load_dotenv
import aiohttp
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

# Импортируем необходимые компоненты LangChain
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    filename='smm_bot.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Инициализация OpenAI API
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set.")
else:
    openai = OpenAI(api_key=openai_api_key)

# Емкость краткосрочной памяти (5-7 сообщений)
SHORT_TERM_MEMORY_SIZE = 7

# Стадии разговора
(
    COLLECT_PRODUCT_INFO,
    COLLECT_USP,
    COLLECT_TARGET_AUDIENCE,
    COLLECT_COMPETITORS,
    COLLECT_BUDGET,
    COLLECT_DURATION,
    GENERATE_REPORT,
) = range(7)

# Инициализация краткосрочной памяти


async def initialize_memory(context):
    if 'message_history' not in context.user_data:
        context.user_data['message_history'] = deque(
            maxlen=SHORT_TERM_MEMORY_SIZE)

# Обработчик команды /start


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await initialize_memory(context)
    await update.message.reply_text(
        "Йоу, коротко опишите ваш продукт или услугу для продвижения в Telegram."
    )
    return COLLECT_PRODUCT_INFO

# Добавление сообщения в краткосрочную память


async def append_to_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_text = update.message.text.strip()
    context.user_data['message_history'].append(message_text)

# Сбор информации о продукте


async def collect_product_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await append_to_memory(update, context)
    context.user_data['product_info'] = update.message.text.strip()
    await update.message.reply_text(
        "Супер! В чем уникальность продукта? Какие преимущества он предлагает по сравнению с конкурентами?"
    )
    return COLLECT_USP

# Сбор уникального торгового предложения


async def collect_usp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await append_to_memory(update, context)
    context.user_data['usp'] = update.message.text.strip()
    await update.message.reply_text(
        "Опиши целевую аудиторию: возраст, пол, интересы, география и другие важные характеристики."
    )
    return COLLECT_TARGET_AUDIENCE

# Сбор информации о целевой аудитории


async def collect_target_audience(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await append_to_memory(update, context)
    context.user_data['target_audience'] = update.message.text.strip()
    await update.message.reply_text(
        "Укажи 1-3 конкурента в Telegram (их @username или ссылки на каналы), с которыми хочешь сравниться или превзойти."
    )
    return COLLECT_COMPETITORS

# Сбор информации о конкурентах


async def collect_competitors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await append_to_memory(update, context)
    context.user_data['competitors'] = update.message.text.strip()
    await update.message.reply_text(
        "Суммарный бюджет на продвижение в Telegram (в тысячах рублей)?"
    )
    return COLLECT_BUDGET

# Сбор информации о бюджете


async def collect_budget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await append_to_memory(update, context)
    context.user_data['budget'] = update.message.text.strip()
    await update.message.reply_text(
        "Длительность маркетинговой кампании (в месяцах)?"
    )
    return COLLECT_DURATION

# Сбор информации о длительности кампании


async def collect_duration(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await append_to_memory(update, context)
    context.user_data['duration'] = update.message.text.strip()
    await update.message.reply_text(
        "Куль! В течение 1 мин. подготовлю верхнеуровневую SMM-стратегию продвижения в Telegram."
    )
    # Запускаем генерацию отчета
    asyncio.create_task(generate_report(update, context, verbose=True))
    return ConversationHandler.END


async def analyze_competitors(competitors):
    # Инициализируем итоговый список для хранения данных о конкурентах
    competitor_data = []

    # Разделяем строку по запятым и очищаем лишние пробелы
    competitor_list = [competitor.strip()
                       for competitor in competitors.split(',')]

    # Добавляем каждого конкурента в итоговый список как словарь
    for competitor in competitor_list:
        # Используем словари
        competitor_data.append(
            {'username': competitor, 'subscribers': 'найди инфо'})

    # Возвращаем итоговый список данных о конкурентах
    return competitor_data


async def generate_report(update: Update, context: ContextTypes.DEFAULT_TYPE, verbose=False):
    try:
        # Получение данных пользователя
        product_info = context.user_data.get('product_info', '')
        usp = context.user_data.get('usp', '')
        target_audience = context.user_data.get('target_audience', '')
        competitors = context.user_data.get('competitors', '')
        budget = context.user_data.get('budget', '')
        duration = context.user_data.get('duration', '')

        # Краткосрочная память
        history_prompt = "\n".join(context.user_data['message_history'])

        # Анализ конкурентов
        competitor_data = await analyze_competitors(competitors)
        logger.info(f"Competitor Data: {competitor_data}")

        # Формирование анализа конкурентов для промпта
        competitor_analysis = "\n".join([
            f"- {comp.get('title', 'найди инфо')} ({comp.get('username', 'найди инфо')}): "
            f"{comp.get('subscribers', 'найди инфо')} подписчиков, средний охват поста {comp.get('avg_post_reach', 'найди инфо')}, "
            for comp in competitor_data if comp
        ])

        # Шаблон промпта для LangChain
        prompt_template = PromptTemplate(
            input_variables=["history", "product_info", "usp",
                             "target_audience", "competitor_analysis", "budget", "duration"],
            template="""
Ты опытный SMM-агент с глубоким пониманием рынка. Используя предоставленную информацию, разработай детализированную SMM-стратегию для продвижения продукта в Telegram.

История общения:
{history}

Описание продукта:
{product_info}

Уникальное торговое предложение (USP):
{usp}

Целевая аудитория:
{target_audience}

Конкуренты и их данные:
{competitor_analysis}

Общий бюджет: {budget} тысяч рублей
Длительность кампании: {duration} месяцев

Требования к стратегии продвижения:
- Подробный контент-план с указанием типов контента и частоты публикаций.
- Рекомендации по каналам продвижения и рекламе.
- Конкретные действия для достижения целей.
- Целевые KPI и метрики успеха с учетом информации выше.

Пожалуйста, представь стратегию в структурированном виде.
"""
        )

        # Инициализация LLMChain с использованием ChatOpenAI
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-4")
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Подготовка входных данных для промпта
        prompt_inputs = {
            "history": history_prompt,
            "product_info": product_info,
            "usp": usp,
            "target_audience": target_audience,
            "competitor_analysis": competitor_analysis,
            "budget": budget,
            "duration": duration
        }

        if verbose:
            full_prompt = prompt_template.format(**prompt_inputs)
            logger.info(f"Prompt for LLM:\n{full_prompt}")

        # Генерация стратегии
        strategy = chain.run(prompt_inputs)

        if verbose:
            logger.info(f"LLM response:\n{strategy}")

        # Пересчет бюджета (пример с использованием встроенного калькулятора)
        budget_allocation = calculate_budget_allocation(
            float(budget), int(duration))

        # Формирование итогового отчета
        report = (
            f"**Твоя SMM Стратегия:**\n\n{strategy}\n\n"
            f"**Стандартное распределение ежемесячного бюджета:**\n{budget_allocation}"
        )
        await update.message.reply_text(report, parse_mode='Markdown')
        logger.info("Report generated and sent to the user.")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        await update.message.reply_text("Произошла ошибка при генерации отчета. Пожалуйста, попробуйте позже.")


def calculate_budget_allocation(budget, duration):
    # Пример простого распределения бюджета
    monthly_budget = budget / duration
    content_creation = monthly_budget * 0.4
    advertising = monthly_budget * 0.5
    analytics = monthly_budget * 0.1

    allocation = (
        f"  - Создание контента {content_creation:,.0f} тыс. рублей\n"
        f"  - Реклама и продвижение {advertising:,.0f} тыс. рублей\n"
        f"  - Аналитика и оптимизация {analytics:,.0f} тыс. рублей\n"
    )
    return allocation

# Главная функция


def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        return

    if not openai_api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return

    application = ApplicationBuilder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            COLLECT_PRODUCT_INFO: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_product_info)],
            COLLECT_USP: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_usp)],
            COLLECT_TARGET_AUDIENCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_target_audience)],
            COLLECT_COMPETITORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_competitors)],
            COLLECT_BUDGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_budget)],
            COLLECT_DURATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_duration)],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    application.add_handler(conv_handler)
    logger.info("Bot is running...")
    application.run_polling()
    logger.info("Bot has stopped.")


if __name__ == '__main__':
    main()
