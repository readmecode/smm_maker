import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
import openai
import requests

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='smm_bot.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# TGStat API configuration
TGSTAT_API_KEY = os.getenv('TGSTAT_API_KEY')  # Ensure you have your TGStat API key
TGSTAT_API_URL = "https://api.tgstat.ru"

# Conversation stages
(
    COLLECT_PRODUCT_INFO,
    COLLECT_USP,
    COLLECT_TARGET_AUDIENCE,
    COLLECT_COMPETITORS,
    COLLECT_CONTENT_TYPE,
    GENERATE_REPORT,
) = range(6)

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Привет! Я ваш SMM Bot. Давайте начнем с описания вашего продукта или услуги для Telegram."
    )
    return COLLECT_PRODUCT_INFO

# Collect product information
async def collect_product_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    product_info = update.message.text.strip()
    context.user_data['product_info'] = product_info
    logger.info(f"Product Info: {product_info}")

    await update.message.reply_text(
        "В чем заключается уникальность вашего продукта? Какие преимущества он предлагает?"
    )
    return COLLECT_USP

# Collect unique selling proposition
async def collect_usp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    usp = update.message.text.strip()
    context.user_data['usp'] = usp
    logger.info(f"USP: {usp}")

    await update.message.reply_text(
        "Опишите вашу целевую аудиторию: возраст, пол, интересы и география."
    )
    return COLLECT_TARGET_AUDIENCE

# Collect target audience details
async def collect_target_audience(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    target_audience = update.message.text.strip()
    context.user_data['target_audience'] = target_audience
    logger.info(f"Target Audience: {target_audience}")

    await update.message.reply_text(
        "Укажите 1-3 конкурента в Telegram (их @username или ссылки на каналы)."
    )
    return COLLECT_COMPETITORS

# Collect competitors information
async def collect_competitors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    competitors = update.message.text.strip()
    context.user_data['competitors'] = competitors
    logger.info(f"Competitors: {competitors}")

    await update.message.reply_text(
        "Спасибо! Теперь я проанализирую данные и подготовлю для вас SMM-стратегию."
    )

    # Start generating report
    asyncio.create_task(generate_report(update, context))
    return ConversationHandler.END

# Generate SMM strategy report
async def generate_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        product_info = context.user_data.get('product_info', '')
        usp = context.user_data.get('usp', '')
        target_audience = context.user_data.get('target_audience', '')
        competitors = context.user_data.get('competitors', '')

        # Analyze competitors using TGStat API
        competitor_data = []
        competitor_list = competitors.split(',')
        for competitor in competitor_list:
            competitor = competitor.strip()
            if competitor.startswith('@'):
                competitor = competitor[1:]
            stats = get_telegram_channel_info(competitor)
            if stats:
                competitor_data.append(stats)
            else:
                competitor_data.append({'username': competitor, 'subscribers': 'Unknown'})
        logger.info(f"Competitor Data: {competitor_data}")

        # Generate strategy using OpenAI
        strategy = await generate_strategy(product_info, usp, target_audience, competitor_data)

        # Estimate budget
        budget = estimate_budget(competitor_data)

        # Send report to user
        report = f"**Ваша SMM-стратегия:**\n\n{strategy}\n\n**Рекомендуемый месячный бюджет на SMM в Telegram:**\n{budget} рублей."
        await update.message.reply_text(report, parse_mode='Markdown')
        logger.info("Report generated and sent to the user.")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        await update.message.reply_text("Произошла ошибка при генерации отчета. Пожалуйста, попробуйте позже.")

# Function to get Telegram channel info using TGStat API
def get_telegram_channel_info(username):
    try:
        url = f"{TGSTAT_API_URL}/channels/search"
        params = {
            'q': username,
            'token': TGSTAT_API_KEY,
            'limit': 1
        }
        response = requests.get(url, params=params)
        data = response.json()
        logger.info(f"TGStat API response for {username}: {data}")

        if 'error' in data:
            logger.error(f"TGStat API error for {username}: {data['error']}")
            return None

        if 'response' in data and data['response']['count'] > 0:
            channel = data['response']['items'][0]
            stats = {
                'username': channel.get('username', ''),
                'title': channel.get('title', ''),
                'subscribers': channel.get('participants_count', 0),
                'avg_post_reach': channel.get('avg_post_reach', 0),
                'err_percent': channel.get('err_percent', 0.0),
            }
            return stats
        else:
            logger.warning(f"No data found for {username}")
            return None
    except Exception as e:
        logger.error(f"Error fetching channel info for {username}: {e}")
        return None

# Function to generate strategy using OpenAI
async def generate_strategy(product_info, usp, target_audience, competitor_data):
    competitor_analysis = "\n".join([
        f"- {comp.get('title', 'Unknown Title')} (@{comp.get('username', 'Unknown')}): "
        f"{comp.get('subscribers', 'Unknown')} подписчиков, средний охват поста {comp.get('avg_post_reach', 'N/A')}, "
        f"ERR {comp.get('err_percent', 'N/A')}%"
        for comp in competitor_data if comp
    ])

    prompt = (
        f"Ты опытный SMM-специалист.\n\n"
        f"**Описание продукта:** {product_info}\n"
        f"**Уникальное торговое предложение (USP):** {usp}\n"
        f"**Целевая аудитория:** {target_audience}\n"
        f"**Конкуренты и их данные:**\n{competitor_analysis}\n\n"
        f"Сформулируй конкретные цели присутствия бизнеса в Telegram и предложи KPI по этим целям. Также предложи конкретные действия по продвижению продукта, учитывая конкурентный анализ.\n"
        f"Представь информацию в структурированном виде."
    )

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,
        temperature=0.0,
    )

    strategy = response['choices'][0]['message']['content']
    return strategy

# Function to estimate budget
def estimate_budget(competitor_data):
    # Estimate budget based on average subscriber count and advertising rates
    total_subscribers = sum([
        comp.get('subscribers', 0) if isinstance(comp.get('subscribers', 0), int) else 0
        for comp in competitor_data
    ])
    avg_subscribers = total_subscribers / len(competitor_data) if competitor_data else 0

    # Assume cost per subscriber acquisition is between 5 to 15 rubles based on engagement and content
    cpa = 10  # Average CPA based on competitor engagement
    estimated_budget_value = avg_subscribers * cpa

    estimated_budget = f"{int(estimated_budget_value):,}".replace(',', ' ')
    return estimated_budget

# Main function
def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        return

    if not openai.api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return

    if not TGSTAT_API_KEY:
        logger.error("TGSTAT_API_KEY is not set.")
        return

    application = ApplicationBuilder().token(token).build()

    # Define conversation handler with the states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            COLLECT_PRODUCT_INFO: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_product_info)],
            COLLECT_USP: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_usp)],
            COLLECT_TARGET_AUDIENCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_target_audience)],
            COLLECT_COMPETITORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_competitors)],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    application.add_handler(conv_handler)

    logger.info("Bot is running...")
    application.run_polling()
    logger.info("Bot has stopped.")

if __name__ == '__main__':
    main()
