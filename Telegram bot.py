"""
Telegram bot front-end for the Text-to-SQL agent/orchestrator project.

Setup:
  1. pip install python-telegram-bot python-dotenv --upgrade
  2. Create a .env file in this same folder with the line:
         TELEGRAM_BOT_TOKEN=your_token_here
     (get the token from @BotFather). Make sure .env is in your .gitignore.
  3. Adjust the import + call in `run_orchestrator()` below to match your actual
     agent1/agent2/orchestrator code.
  4. Run: python telegram_bot.py

This uses polling (not webhooks), so it works from any PC with just an internet
connection - no need to open ports, get a static IP, or host a public server.
As long as this script keeps running, anyone who messages the bot on Telegram
gets a response.
"""

import os
import logging

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Real orchestrator wiring.
#    Assumes orchestrator.py sits at the project root, next to agent1/ and
#    agent2/. If your orchestrator file has a different name, change the
#    import below to match.
# ---------------------------------------------------------------------------
from orchestrator import (
    AGENTS,
    AGENTS_BY_ID,
    get_agent_dataset,
    route_question,
    answer_question,
)
import agent1.main as agent1_core  # resolve_follow_up lives here (shared util)

# One conversation history per Telegram chat, so follow-up questions like
# "what about their wages?" resolve correctly per-user, and separate users
# messaging the bot don't bleed into each other's context. This lives only
# in memory - it resets if the bot script restarts.
_HISTORIES: dict[int, list] = {}

# Build each agent's database once, at import time, so the first real
# message doesn't stall while agent1/agent2 build their SQLite DBs.
for _agent in AGENTS:
    get_agent_dataset(_agent)


def run_orchestrator(user_message: str, chat_id: int) -> str:
    """Routes a Telegram message through the same pipeline as the terminal
    version of the orchestrator (resolve follow-up -> route -> answer),
    keeping history per chat_id."""
    history = _HISTORIES.setdefault(chat_id, [])

    standalone = agent1_core.resolve_follow_up(user_message, history)
    agent_id, reasoning = route_question(standalone, AGENTS, history)
    agent = AGENTS_BY_ID[agent_id]
    dataset = get_agent_dataset(agent)

    answer_text, charts, sql = answer_question(standalone, agent, dataset)

    history.append({
        "role": "user",
        "content": f"{user_message} (interpreted as: {standalone})",
    })
    assistant_turn = {
        "role": "assistant",
        "content": f"[Answered by {agent['name']}] {answer_text}",
    }
    if sql:
        assistant_turn["sql"] = sql
    history.append(assistant_turn)

    # NOTE: `charts` from multi_query_analysis are ignored here for now -
    # Telegram needs images sent separately via bot.send_photo(), not as
    # part of the text reply. Ask if you want that wired up too.
    return answer_text


# ---------------------------------------------------------------------------
# 2. Telegram handlers - shouldn't need much editing.
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a question and I'll route it through the agents."
    )


TELEGRAM_MAX_LEN = 4096  # Telegram's hard per-message character limit
MAX_TABLE_ROWS = 6       # keep answers short/scannable on a phone screen
import html as _html
import re as _re


def _format_markdown_table(table_lines: list) -> str:
    """Turn a block of markdown '| a | b |' lines into a padded, aligned
    plain-text table wrapped for <pre>, trimmed to MAX_TABLE_ROWS data rows."""
    rows = []
    for line in table_lines:
        line = line.strip().strip("|")
        if _re.fullmatch(r"[\s:\-|]+", line):
            continue  # the '---|---' separator row - skip it
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
    """Converts the orchestrator's markdown-ish answer into something that
    actually renders well in Telegram: ### headers -> bold, **bold** kept as
    HTML bold, pipe tables -> padded monospace blocks (trimmed to
    MAX_TABLE_ROWS), and stray '---' divider lines dropped."""
    out_lines = []
    i = 0
    lines = text.split("\n")
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Collect a run of consecutive '| ... |' lines into one table block.
        if stripped.startswith("|"):
            table_block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_block.append(lines[i])
                i += 1
            out_lines.append(_format_markdown_table(table_block))
            continue

        if _re.fullmatch(r"-{3,}", stripped):
            i += 1
            continue  # drop standalone '---' divider lines

        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            out_lines.append(f"<b>{_html.escape(heading)}</b>")
            i += 1
            continue

        # Inline **bold** -> <b>bold</b>; escape everything else for HTML mode.
        escaped = _html.escape(line)
        escaped = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        out_lines.append(escaped)
        i += 1

    # Collapse 3+ blank lines down to a single blank line for compactness.
    joined = "\n".join(out_lines)
    return _re.sub(r"\n{3,}", "\n\n", joined).strip()


async def send_long_message(update: Update, text: str) -> None:
    """Telegram rejects any single message over ~4096 characters (analytical
    answers with full tables can easily exceed this). Split on line breaks
    where possible so tables don't get cut mid-row, but if a single line is
    itself longer than the limit (e.g. one huge unbroken row), hard-slice it
    too - otherwise that one line alone would still fail to send."""
    text = format_for_telegram(text)

    chunks = []
    chunk = ""
    for line in text.split("\n"):
        # A single line longer than the whole limit has to be hard-sliced on
        # its own, independent of everything else.
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    chat_id = update.effective_chat.id
    logger.info("Received from %s: %s", chat_id, user_message)

    # Immediate feedback - analytical questions can take a long time with a
    # local model, and without this the bot looks frozen/unresponsive.
    await update.message.reply_text("Working on it, this may take a while for analysis questions...")

    try:
        # run_orchestrator is synchronous and can be slow (many sequential
        # LLM calls for analytical questions). Running it in a thread keeps
        # the bot's event loop free to keep polling/responding to others
        # while this chat's answer is being computed.
        import asyncio
        reply_text = await asyncio.to_thread(run_orchestrator, user_message, chat_id)
    except Exception as exc:  # keep the bot alive even if the orchestrator errors
        logger.exception("Orchestrator raised an exception")
        reply_text = f"Sorry, something went wrong: {exc}"

    try:
        await send_long_message(update, reply_text)
    except Exception:
        logger.exception("Failed to send reply back to Telegram")
        await update.message.reply_text(
            "I got an answer but couldn't send it back (it may be malformed). Check the server logs."
        )


def main() -> None:
    load_dotenv()  # reads .env in the current working directory into os.environ

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Create a .env file next to this "
            "script containing TELEGRAM_BOT_TOKEN=your_token_here "
            "(get the token from @BotFather on Telegram)."
        )

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is starting (polling mode)...")
    app.run_polling()


if __name__ == "__main__":
    main()