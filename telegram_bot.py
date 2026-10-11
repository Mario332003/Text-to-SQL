"""
Telegram bot front-end for the Text-to-SQL agent/orchestrator project.

Supports regular text questions plus per-chat CSV/XLSX/XLS attachments. Each
chat's latest uploaded file is converted into a temporary SQLite dataset and is
available to the orchestrator as an "attachment" agent alongside ecommerce,
FIFA, and weather.
"""

import asyncio
import html as _html
import logging
import os
import re as _re
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from orchestrator import (
    AGENTS,
    create_attachment_agent,
    get_agent_dataset,
    run_image_question,
    run_multi_agent_question,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

ATTACHMENTS_DIR = BASE_DIR / "attachments"
IMAGE_UPLOADS_DIR = BASE_DIR / "image_uploads"
ALLOWED_ATTACHMENT_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# One conversation history and one latest uploaded file per Telegram chat.
_HISTORIES: dict[int, list] = {}
_CHAT_ATTACHMENTS: dict[int, dict] = {}

# Permanent agent datasets are built lazily on the first relevant question.
# That lets Telegram polling start immediately instead of appearing frozen while
# local SQLite/schema setup or Ollama work happens during import.


# ---------------------------------------------------------------------------
# Attachment preparation
# ---------------------------------------------------------------------------
def _safe_file_name(name: str) -> str:
    name = Path(name or "attachment").name
    cleaned = _re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "attachment"


def _attachment_folder(chat_id: int) -> Path:
    folder = ATTACHMENTS_DIR / str(chat_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _image_folder(chat_id: int) -> Path:
    folder = IMAGE_UPLOADS_DIR / str(chat_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _prepared_csv_path(uploaded_path: Path) -> Path:
    return uploaded_path.with_suffix(".prepared.csv")


def _convert_attachment_to_csv(uploaded_path: Path) -> Path:
    ext = uploaded_path.suffix.lower()
    csv_path = _prepared_csv_path(uploaded_path)

    if ext == ".csv":
        # Keep a stable prepared copy; the database builder fingerprints this file.
        shutil.copyfile(uploaded_path, csv_path)
        return csv_path

    if ext in {".xlsx", ".xls"}:
        frame = pd.read_excel(uploaded_path, sheet_name=0)
        if frame.empty and len(frame.columns) == 0:
            raise ValueError("The spreadsheet does not contain a header row or data.")
        frame.to_csv(csv_path, index=False)
        return csv_path

    raise ValueError("Please upload a CSV, XLSX, or XLS file.")


def prepare_chat_attachment(chat_id: int, uploaded_path: Path, original_name: str) -> dict:
    csv_path = _convert_attachment_to_csv(uploaded_path)
    dataset = AGENTS[0]["core"].create_database(str(csv_path))
    attachment = {
        "file_name": original_name,
        "file_path": str(uploaded_path),
        "csv_path": str(csv_path),
        "db_path": dataset["db_path"],
        "schema_path": dataset["schema_path"],
        "dataset": dataset,
    }
    _CHAT_ATTACHMENTS[chat_id] = attachment
    return attachment


def agents_for_chat(chat_id: int) -> list:
    agents = list(AGENTS)
    attachment = _CHAT_ATTACHMENTS.get(chat_id)
    if attachment:
        agents.append(create_attachment_agent(attachment["dataset"], attachment["file_name"]))
    return agents


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------
def run_orchestrator(user_message: str, chat_id: int) -> str:
    history = _HISTORIES.setdefault(chat_id, [])
    result = run_multi_agent_question(user_message, agents_for_chat(chat_id), history)

    routed_names = ", ".join(agent["name"] for agent in result["routed_agents"])
    history.append({
        "role": "user",
        "content": f"{user_message} (interpreted as: {result['standalone_question']})",
    })
    assistant_turn = {
        "role": "assistant",
        "content": f"[Answered by {routed_names}] {result['text']}",
    }
    if result.get("sql"):
        assistant_turn["sql"] = result["sql"]
    history.append(assistant_turn)

    return result["text"]


def run_architecture_image_orchestrator(image_path: Path, chat_id: int, caption: str = "") -> dict:
    history = _HISTORIES.setdefault(chat_id, [])
    return run_image_question(str(image_path), caption or None, history)


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------
TELEGRAM_MAX_LEN = 4096
MAX_TABLE_ROWS = 6


def _format_markdown_table(table_lines: list) -> str:
    rows = []
    for line in table_lines:
        line = line.strip().strip("|")
        if _re.fullmatch(r"[\s:\-|]+", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        rows.append(cells)
    if not rows:
        return ""

    header, data_rows = rows[0], rows[1:]
    truncated = len(data_rows) > MAX_TABLE_ROWS
    shown_rows = data_rows[:MAX_TABLE_ROWS]

    col_widths = [len(h) for h in header]
    for row in shown_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(row) if i < len(col_widths))

    lines_out = [fmt_row(header), "-+-".join("-" * w for w in col_widths)]
    lines_out += [fmt_row(r) for r in shown_rows]
    if truncated:
        lines_out.append(f"... (+{len(data_rows) - MAX_TABLE_ROWS} more rows)")

    return "<pre>" + _html.escape("\n".join(lines_out)) + "</pre>"


def format_for_telegram(text: str) -> str:
    out_lines = []
    i = 0
    lines = text.split("\n")
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("|"):
            table_block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_block.append(lines[i])
                i += 1
            out_lines.append(_format_markdown_table(table_block))
            continue

        if _re.fullmatch(r"-{3,}", stripped):
            i += 1
            continue

        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            out_lines.append(f"<b>{_html.escape(heading)}</b>")
            i += 1
            continue

        escaped = _html.escape(line)
        escaped = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        out_lines.append(escaped)
        i += 1

    joined = "\n".join(out_lines)
    return _re.sub(r"\n{3,}", "\n\n", joined).strip()


async def send_long_message(update: Update, text: str) -> None:
    text = format_for_telegram(text)
    chunks = []
    chunk = ""
    for line in text.split("\n"):
        if len(line) > TELEGRAM_MAX_LEN:
            if chunk:
                chunks.append(chunk)
                chunk = ""
            for i in range(0, len(line), TELEGRAM_MAX_LEN):
                chunks.append(line[i:i + TELEGRAM_MAX_LEN])
            continue

        if len(chunk) + len(line) + 1 > TELEGRAM_MAX_LEN:
            chunks.append(chunk)
            chunk = line
        else:
            chunk = f"{chunk}\n{line}" if chunk else line
    if chunk:
        chunks.append(chunk)

    for c in chunks:
        await update.message.reply_text(c, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram bot error", exc_info=context.error)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
       """ 👋 Hi! I’m a multi-agent AI assistant.

I can help you with:

🛒 E-Commerce Data — Ask about customers, orders, sales, products, and other business insights.

⚽ FIFA Data — Ask about players, teams, ratings, nationalities, positions, and football statistics.

🌤️ Live Weather — Ask for current or forecast weather information for locations around the world.

🏠 Architecture Sketches — Send a floor-plan or villa exterior sketch and I’ll create a clean architectural output.

🔗 Composite Questions — Combine information from multiple sources in a single question.

📎 You can also upload a CSV, XLSX, or XLS file and ask questions that combine your file with the available data."""


    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    document = update.message.document
    original_name = _safe_file_name(document.file_name or "attachment")
    ext = Path(original_name).suffix.lower()

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        folder = _image_folder(chat_id)
        uploaded_path = folder / f"{document.file_unique_id}_{original_name}"
        await update.message.reply_text("Got the sketch. Analyzing it and creating the architectural output...")
        try:
            telegram_file = await context.bot.get_file(document.file_id)
            await telegram_file.download_to_drive(custom_path=str(uploaded_path))
            result = await asyncio.to_thread(
                run_architecture_image_orchestrator,
                uploaded_path,
                chat_id,
                (update.message.caption or "").strip(),
            )
        except Exception as exc:
            logger.exception("Failed to convert image attachment")
            await update.message.reply_text(f"Sorry, I couldn't convert that sketch: {exc}")
            return

        await _send_architecture_image_result(update, result)
        return

    if ext not in ALLOWED_ATTACHMENT_EXTENSIONS:
        await update.message.reply_text("Please upload a CSV, XLSX, XLS, JPG, PNG, or WEBP file.")
        return

    folder = _attachment_folder(chat_id)
    uploaded_path = folder / f"{document.file_unique_id}_{original_name}"

    await update.message.reply_text("Got the file. Preparing it for questions...")
    try:
        telegram_file = await context.bot.get_file(document.file_id)
        await telegram_file.download_to_drive(custom_path=str(uploaded_path))
        attachment = await asyncio.to_thread(
            prepare_chat_attachment,
            chat_id,
            uploaded_path,
            original_name,
        )
    except Exception as exc:
        logger.exception("Failed to prepare attachment")
        await update.message.reply_text(f"Sorry, I couldn't prepare that file: {exc}")
        return

    await update.message.reply_text(
        f"Ready: {attachment['file_name']}. Ask a question about this file, "
        "or compare it with ecommerce, FIFA, or weather data."
    )

    caption = (update.message.caption or "").strip()
    if caption:
        await _answer_text(update, caption, chat_id)


async def _answer_text(update: Update, user_message: str, chat_id: int) -> None:
    logger.info("Received from %s: %s", chat_id, user_message)
    await update.message.reply_text("Working on it, this may take a while for analysis questions...")

    try:
        reply_text = await asyncio.to_thread(run_orchestrator, user_message, chat_id)
    except Exception as exc:
        logger.exception("Orchestrator raised an exception")
        reply_text = f"Sorry, something went wrong: {exc}"

    try:
        await send_long_message(update, reply_text)
    except Exception:
        logger.exception("Failed to send reply back to Telegram")
        await update.message.reply_text(
            "I got an answer but couldn't send it back. Check the server logs."
        )


async def _send_architecture_image_result(update: Update, result: dict) -> None:
    image_path = result.get("image_path")
    caption = result.get("text") or "Here is the generated architectural image."
    if not image_path or not Path(image_path).exists():
        await update.message.reply_text(caption)
        return
    with open(image_path, "rb") as image_file:
        await update.message.reply_photo(photo=image_file, caption=caption)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    photo = update.message.photo[-1]
    folder = _image_folder(chat_id)
    uploaded_path = folder / f"{photo.file_unique_id}.jpg"

    await update.message.reply_text("Got the sketch. Analyzing it and creating the architectural output...")
    try:
        telegram_file = await context.bot.get_file(photo.file_id)
        await telegram_file.download_to_drive(custom_path=str(uploaded_path))
        result = await asyncio.to_thread(
            run_architecture_image_orchestrator,
            uploaded_path,
            chat_id,
            (update.message.caption or "").strip(),
        )
    except Exception as exc:
        logger.exception("Failed to convert photo")
        await update.message.reply_text(f"Sorry, I couldn't convert that sketch: {exc}")
        return

    await _send_architecture_image_result(update, result)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    chat_id = update.effective_chat.id
    await _answer_text(update, user_message, chat_id)


def main() -> None:
    load_dotenv()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Create a .env file next to this "
            "script containing TELEGRAM_BOT_TOKEN=your_token_here."
        )

    app = ApplicationBuilder().token(token).build()
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting (polling mode)...")
    app.run_polling()


if __name__ == "__main__":
    main()
