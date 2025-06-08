#!/usr/bin/env python3
# hypebot.py – исправленная версия

"""
Исправления:
- Добавлена проверка валидности изображений перед отправкой
- Улучшена обработка ошибок и timeout'ов
- Исправлена логика создания media groups
- Добавлена проверка размера callback_data
- Улучшена обработка HTML-контента
- Добавлена защита от дублирования постов
"""

import os
import json
import logging
import hashlib
import asyncio
import warnings
import httpx
import openai
from bs4 import BeautifulSoup, FeatureNotFound, XMLParsedAsHTMLWarning
from telegram import InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
from telegram.error import TelegramError, Conflict
from urllib.parse import urljoin, urlparse

# Подавляем предупреждения BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- Настройка ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Переменные окружения ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHANNEL = os.getenv("TELEGRAM_CHAT_ID")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STATE_FILE = "state.json"
CHECK_INTERVAL_SECONDS = 1800  # 30 минут

# --- Источники релизов ---
SOURCES = [
    {"key": "sneakernews", "name": "SneakerNews", "type": "json", "api": "https://sneakernews.com/wp-json/wp/v2/posts?per_page=5&_embed"},
    {"key": "hypebeast", "name": "Hypebeast Footwear", "type": "rss", "api": "https://hypebeast.com/footwear/feed/"},
    {"key": "highsnobiety", "name": "Highsnobiety", "type": "rss", "api": "https://www.highsnobiety.com/category/sneakers/feed/"},
]

# --- Проверка конфигурации ---
if not all([TELEGRAM_TOKEN, TELEGRAM_CHANNEL, ADMIN_CHAT_ID, OPENAI_API_KEY]):
    logging.critical("Не заданы обязательные переменные окружения")
    exit(1)

try:
    ADMIN_CHAT_ID = int(ADMIN_CHAT_ID)
except ValueError:
    logging.critical("ADMIN_CHAT_ID должен быть числом")
    exit(1)

if not str(TELEGRAM_CHANNEL).startswith("@") and not str(TELEGRAM_CHANNEL).startswith("-"):
    try:
        TELEGRAM_CHANNEL = int(TELEGRAM_CHANNEL)
    except ValueError:
        logging.critical("TELEGRAM_CHAT_ID должен быть числом или начинаться с @")
        exit(1)

# --- Инициализация клиентов и состояния ---
openai.api_key = OPENAI_API_KEY

try:
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
        # Проверяем структуру состояния
        if not isinstance(state.get("sent_links"), list):
            state["sent_links"] = []
        if not isinstance(state.get("pending"), dict):
            state["pending"] = {}
            
        # Очищаем некорректные записи в pending
        valid_pending = {}
        for uid, record in state["pending"].items():
            if isinstance(record, dict) and all(key in record for key in ['id', 'title', 'link']):
                valid_pending[uid] = record
            else:
                logging.warning(f"Удаляю некорректную запись из pending: {uid}")
        state["pending"] = valid_pending
        
except (FileNotFoundError, json.JSONDecodeError):
    state = {"sent_links": [], "pending": {}}

def save_state():
    """Сохраняет состояние в файл с обработкой ошибок."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Ошибка при сохранении состояния: {e}")

def make_id(source: str, link: str) -> str:
    """Создает уникальный ID для поста."""
    return hashlib.md5(f"{source}|{link}".encode()).hexdigest()[:12]

def is_valid_image_url(url: str) -> bool:
    """Проверяет, является ли URL валидным изображением."""
    if not url or not isinstance(url, str):
        return False
    
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    
    # Проверяем расширение файла
    path = parsed.path.lower()
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    return any(path.endswith(ext) for ext in valid_extensions)

async def validate_image_url(client: httpx.AsyncClient, url: str) -> bool:
    """Проверяет доступность изображения."""
    try:
        response = await client.head(url, timeout=10)
        return (response.status_code == 200 and 
                response.headers.get('content-type', '').startswith('image/'))
    except:
        return False

async def fetch_releases(client: httpx.AsyncClient) -> list:
    """Асинхронно получает новые релизы из всех источников."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    releases = []
    
    for src in SOURCES:
        try:
            logging.info(f"Проверяю источник: {src['name']}")
            resp = await client.get(src["api"], headers=headers, timeout=20)
            resp.raise_for_status()

            if src["type"] == "json":
                try:
                    posts = resp.json()
                    if not isinstance(posts, list):
                        logging.warning(f"Неожиданный формат данных от {src['name']}")
                        continue
                except json.JSONDecodeError:
                    logging.error(f"Ошибка парсинга JSON от {src['name']}")
                    continue

                for post in posts[:5]:  # Ограничиваем количество постов
                    try:
                        link = post.get("link")
                        title_data = post.get("title", {})
                        title = title_data.get("rendered", "") if isinstance(title_data, dict) else str(title_data)
                        title = BeautifulSoup(title, "html.parser").get_text(strip=True)
                        
                        if not link or not title or len(title) < 10:
                            continue

                        uid = make_id(src["key"], link)
                        if uid in state["pending"] or link in state["sent_links"]:
                            continue

                        # Получаем изображения
                        images = []
                        media = post.get("_embedded", {}).get("wp:featuredmedia", [])
                        if media and isinstance(media, list) and len(media) > 0:
                            featured_url = media[0].get("source_url")
                            if featured_url and is_valid_image_url(featured_url):
                                images.append(featured_url)
                        
                        # Получаем дополнительный контент
                        context = ""
                        try:
                            page_resp = await client.get(link, headers=headers, timeout=15)
                            if page_resp.status_code == 200:
                                soup = BeautifulSoup(page_resp.text, "html.parser")
                                
                                # Дополнительные изображения
                                for img in soup.select("div.sn-gallery-wrapper img[src]")[:3]:
                                    img_url = img.get("src")
                                    if img_url and is_valid_image_url(img_url) and img_url not in images:
                                        images.append(img_url)
                                
                                # Контент статьи
                                content_div = soup.select_one("div.article-with-ad-col.article-left-content")
                                if content_div:
                                    paragraphs = []
                                    for p in content_div.find_all("p")[:3]:  # Ограничиваем количество параграфов
                                        text = p.get_text(strip=True)
                                        if text and len(text) > 20 and not p.find_parent("div", class_="advertisement-section"):
                                            paragraphs.append(text)
                                    context = "\n\n".join(paragraphs)
                        except Exception as e:
                            logging.warning(f"Не удалось получить контент страницы {link}: {e}")
                        
                        # Валидируем изображения
                        valid_images = []
                        for img_url in images[:5]:  # Ограничиваем количество изображений
                            if await validate_image_url(client, img_url):
                                valid_images.append(img_url)
                            await asyncio.sleep(0.1)  # Небольшая задержка
                        
                        releases.append({
                            "id": uid,
                            "title": title[:200],  # Ограничиваем длину заголовка
                            "link": link,
                            "images": valid_images,
                            "context": context[:1000]  # Ограничиваем длину контекста
                        })
                        
                        # Отладочная информация
                        logging.debug(f"Создан релиз: id={uid}, title={title[:50]}, images_count={len(valid_images)}")
                        
                    except Exception as e:
                        logging.error(f"Ошибка при обработке поста: {e}")
                        continue

            elif src["type"] == "rss":
                try:
                    soup = BeautifulSoup(resp.text, "xml")
                except FeatureNotFound:
                    soup = BeautifulSoup(resp.text, "html.parser")

                for item in soup.select("item")[:5]:
                    try:
                        link_elem = item.find("link")
                        title_elem = item.find("title")
                        
                        if not link_elem or not title_elem:
                            continue
                            
                        link = link_elem.get_text(strip=True)
                        title = title_elem.get_text(strip=True)
                        
                        if not link or not title or len(title) < 10:
                            continue

                        uid = make_id(src["key"], link)
                        if uid in state["pending"] or link in state["sent_links"]:
                            continue
                        
                        # Для RSS можно добавить парсинг изображений
                        images = []
                        description = ""
                        
                        desc_elem = item.find("description")
                        if desc_elem:
                            desc_soup = BeautifulSoup(desc_elem.get_text(), "html.parser")
                            description = desc_soup.get_text(strip=True)[:500]
                        
                        releases.append({
                            "id": uid,
                            "title": title[:200],
                            "link": link,
                            "images": images,
                            "context": description
                        })
                        
                    except Exception as e:
                        logging.error(f"Ошибка при обработке RSS-элемента: {e}")
                        continue

        except httpx.TimeoutException:
            logging.error(f"Timeout при запросе к {src['name']}")
        except httpx.RequestError as e:
            logging.error(f"Ошибка HTTP при запросе к {src['name']}: {e}")
        except Exception as e:
            logging.error(f"Неожиданная ошибка при обработке {src['name']}: {e}")
            
    logging.info(f"Найдено {len(releases)} новых релизов")
    return releases

async def gen_caption(title: str, context: str) -> str:
    """Генерирует текст поста с помощью GPT."""
    system_prompt = """Ты — автор Telegram-канала о кроссовках и streetwear. 
    Создай короткий (до 300 символов), стильный пост на русском языке.
    Используй эмодзи и современный сленг. Формат: заголовок + краткое описание + призыв к действию."""
    
    user_prompt = f"Заголовок: {title}\nДетали: {context[:500]}"
    
    try:
        # Используем синхронный вызов для совместимости со старой версией
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Используем gpt-3.5-turbo для совместимости
                temperature=0.7,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
        )
        generated = response.choices[0].message.content.strip()
        return generated if generated else f"🔥 **{title}**\n\nНовый релиз уже скоро!"
    except Exception as e:
        logging.error(f"Ошибка при обращении к OpenAI: {e}")
        return f"🔥 **{title}**\n\nСкоро в продаже! Не пропустите этот релиз."

def build_media_group(record: dict, for_channel: bool = False) -> list:
    """Собирает медиа-группу для отправки."""
    if not record.get("images"):
        return []
    
    caption = record['description']
    if for_channel and len(caption) + len(record['link']) + 20 < 1024:
        caption += f"\n\n[Подробнее]({record['link']})"
    
    # Ограничиваем длину caption (Telegram лимит - 1024 символа)
    if len(caption) > 1000:
        caption = caption[:997] + "..."
    
    media = []
    # Первое изображение с подписью
    media.append(InputMediaPhoto(
        media=record["images"][0], 
        caption=caption,
        parse_mode=ParseMode.MARKDOWN
    ))
    
    # Остальные изображения без подписи (максимум 9, так как первое уже добавлено)
    for url in record["images"][1:9]:
        media.append(InputMediaPhoto(media=url))
    
    return media

async def send_for_moderation(bot, record: dict):
    """Отправляет пост на модерацию админу."""
    # Проверяем наличие обязательных полей
    if not isinstance(record, dict):
        logging.error("Record не является словарем")
        return False
        
    required_fields = ['id', 'title', 'description', 'link']
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        logging.error(f"Отсутствуют обязательные поля в record: {missing_fields}")
        logging.error(f"Содержимое record: {record}")
        return False
    
    logging.info(f"Отправляю на модерацию: {record['title'][:50]}...")
    
    try:
        # Проверяем длину callback_data (Telegram лимит - 64 байта)
        approve_data = f"approve:{record['id']}"
        reject_data = f"reject:{record['id']}"
        
        if len(approve_data.encode()) > 64 or len(reject_data.encode()) > 64:
            logging.error(f"Callback data слишком длинный для {record['id']}")
            return False
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Опубликовать", callback_data=approve_data)],
            [InlineKeyboardButton("❌ Пропустить", callback_data=reject_data)],
        ])

        if record.get("images"):
            media = build_media_group(record, for_channel=False)
            if media:
                await bot.send_media_group(ADMIN_CHAT_ID, media)
                await bot.send_message(
                    ADMIN_CHAT_ID, 
                    f"Выберите действие для поста:\n**{record['title']}**", 
                    reply_markup=keyboard,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                # Если нет валидных изображений, отправляем текстом
                text = f"**{record['title']}**\n\n{record['description']}\n\n[Подробнее]({record['link']})"
                await bot.send_message(
                    ADMIN_CHAT_ID, 
                    text, 
                    parse_mode=ParseMode.MARKDOWN, 
                    reply_markup=keyboard
                )
        else:
            text = f"**{record['title']}**\n\n{record['description']}\n\n[Подробнее]({record['link']})"
            await bot.send_message(
                ADMIN_CHAT_ID, 
                text, 
                parse_mode=ParseMode.MARKDOWN, 
                reply_markup=keyboard
            )
        return True
        
    except TelegramError as e:
        logging.error(f"Ошибка Telegram при отправке на модерацию: {e}")
        return False
    except Exception as e:
        logging.error(f"Неожиданная ошибка при отправке на модерацию: {e}")
        logging.error(f"Record content: {record}")
        return False

async def publish_release(bot, record: dict):
    """Публикует пост в основной канал."""
    logging.info(f"Публикую в канал: {record['title'][:50]}...")
    
    try:
        if record.get("images"):
            media = build_media_group(record, for_channel=True)
            if media:
                await bot.send_media_group(TELEGRAM_CHANNEL, media)
            else:
                # Если нет валидных изображений, отправляем текстом
                text = f"**{record['title']}**\n\n{record['description']}\n\n[Подробнее]({record['link']})"
                await bot.send_message(
                    TELEGRAM_CHANNEL, 
                    text, 
                    parse_mode=ParseMode.MARKDOWN
                )
        else:
            text = f"**{record['title']}**\n\n{record['description']}\n\n[Подробнее]({record['link']})"
            await bot.send_message(
                TELEGRAM_CHANNEL, 
                text, 
                parse_mode=ParseMode.MARKDOWN
            )
        return True
        
    except TelegramError as e:
        logging.error(f"Ошибка Telegram при публикации: {e}")
        return False
    except Exception as e:
        logging.error(f"Неожиданная ошибка при публикации: {e}")
        return False

# --- Обработчики Telegram ---

async def check_releases_job(context: ContextTypes.DEFAULT_TYPE):
    """Основная задача, выполняемая по расписанию."""
    bot = context.bot
    
    try:
        # 1. Повторная отправка "зависших" постов
        pending_items = list(state["pending"].items())
        if pending_items:
            logging.info(f"Найдено {len(pending_items)} постов в ожидании")
            for uid, record in pending_items[:3]:  # Ограничиваем количество повторных отправок
                # Проверяем, что record содержит все необходимые поля
                if not isinstance(record, dict) or 'id' not in record:
                    logging.error(f"Некорректные данные в pending для {uid}: {record}")
                    state["pending"].pop(uid, None)
                    save_state()
                    continue
                    
                success = await send_for_moderation(bot, record)
                if not success:
                    logging.error(f"Не удалось отправить на модерацию: {record.get('link', uid)}")
                await asyncio.sleep(2)

        # 2. Поиск новых релизов
        logging.info("Ищу новые релизы...")
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            new_releases = await fetch_releases(client)

        if not new_releases:
            logging.info("Новых релизов не найдено")
            return

        logging.info(f"Найдено {len(new_releases)} новых релизов")
        
        # 3. Обработка новых релизов
        for rel in new_releases[:5]:  # Ограничиваем количество новых постов за раз
            try:
                description = await gen_caption(rel["title"], rel["context"])
                record = {**rel, "description": description}
                
                state["pending"][rel["id"]] = record
                save_state()
                
                success = await send_for_moderation(bot, record)
                if not success:
                    logging.error(f"Не удалось отправить на модерацию: {rel.get('link', rel.get('id', 'unknown'))}")
                    # Удаляем из pending, если не удалось отправить
                    state["pending"].pop(rel["id"], None)
                    save_state()
                
                await asyncio.sleep(3)  # Задержка между постами
                
            except Exception as e:
                logging.error(f"Ошибка при обработке релиза {rel.get('link', rel.get('id', 'unknown'))}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Ошибка в check_releases_job: {e}")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатия на инлайн-кнопки."""
    query = update.callback_query
    await query.answer()

    try:
        if ":" not in query.data:
            await query.edit_message_text("❌ Ошибка: некорректный формат данных")
            return
            
        action, uid = query.data.split(":", 1)
        
        if action not in ["approve", "reject"]:
            await query.edit_message_text("❌ Ошибка: неизвестное действие")
            return
            
    except Exception as e:
        logging.error(f"Ошибка при парсинге callback_data: {e}")
        await query.edit_message_text("❌ Ошибка при обработке действия")
        return

    record = state["pending"].get(uid)
    if not record:
        await query.edit_message_text("❌ Этот пост уже был обработан")
        return

    try:
        if action == "approve":
            published = await publish_release(context.bot, record)
            if published:
                await query.edit_message_text(f"✅ Опубликовано: {record['title'][:50]}...")
                state["sent_links"].append(record["link"])
                # Ограничиваем размер списка отправленных ссылок
                if len(state["sent_links"]) > 1000:
                    state["sent_links"] = state["sent_links"][-500:]
            else:
                await query.edit_message_text(f"🚨 Ошибка публикации: {record['title'][:50]}...")
                return  # Не удаляем из pending

        elif action == "reject":
            await query.edit_message_text(f"❌ Пропущено: {record['title'][:50]}...")

        # Удаляем из очереди
        state["pending"].pop(uid, None)
        save_state()
        
    except Exception as e:
        logging.error(f"Ошибка при обработке callback: {e}")
        await query.edit_message_text("❌ Произошла ошибка при обработке")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start."""
    await update.message.reply_text(
        "👟 Привет! Я HypeBot для мониторинга релизов кроссовок.\n\n"
        "Команды:\n"
        "/check - проверить новые релизы (только для админа)\n"
        "/status - показать статус бота"
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /status."""
    pending_count = len(state["pending"])
    sent_count = len(state["sent_links"])
    
    status_text = (
        f"📊 Статус бота:\n\n"
        f"📝 Постов в ожидании: {pending_count}\n"
        f"✅ Опубликовано всего: {sent_count}\n"
        f"🔄 Интервал проверки: {CHECK_INTERVAL_SECONDS // 60} минут"
    )
    
    await update.message.reply_text(status_text)

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /check."""
    user_id = update.message.from_user.id
    if user_id != ADMIN_CHAT_ID:
        await update.message.reply_text("❌ Эта команда доступна только администратору")
        return
        
    await update.message.reply_text("🔄 Запускаю проверку новых релизов...")
    
    # Запускаем задачу в фоне
    asyncio.create_task(check_releases_job(context))

def main() -> None:
    """Запускает бота."""
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()

        # Команды
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("check", check_command))
        app.add_handler(CommandHandler("status", status_command))

        # Обработчик кнопок
        app.add_handler(CallbackQueryHandler(on_callback))

        # Периодическая задача
        app.job_queue.run_repeating(
            check_releases_job,
            interval=CHECK_INTERVAL_SECONDS,
            first=30  # Первый запуск через 30 секунд
        )

        logging.info("=== HypeBot запущен ===")
        try:
            app.run_polling(drop_pending_updates=True)
        except Conflict:
            logging.critical("Бот уже запущен в другом процессе. Завершаю работу.")
            return
        
    except Exception as e:
        logging.critical(f"Критическая ошибка при запуске бота: {e}")
        exit(1)

if __name__ == "__main__":
    main()
