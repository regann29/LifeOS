import logging
import os
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from core.hosted import is_hosted
from core.llm import stream_strict_answer
from briefs.service import run_morning_brief

logger = logging.getLogger(__name__)


def run_scheduled_brief() -> None:
    try:
        run_morning_brief(stream_strict_answer)
    except Exception as error:
        logger.exception("Scheduled morning brief failed (%s)", type(error).__name__)


def create_scheduler() -> BackgroundScheduler | None:
    if is_hosted():
        return None
    enabled = os.getenv("BRIEF_ENABLED", "false").strip().casefold()
    if enabled not in {"true", "1", "yes", "on", "false", "0", "no", "off"}:
        raise RuntimeError("BRIEF_ENABLED must be true or false.")
    if enabled not in {"true", "1", "yes", "on"}:
        return None
    raw_time = os.getenv("BRIEF_TIME", "08:00")
    try:
        scheduled_time = datetime.strptime(raw_time, "%H:%M")
    except ValueError as error:
        raise RuntimeError("BRIEF_TIME must use 24-hour HH:MM format.") from error
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_scheduled_brief,
        "cron",
        hour=scheduled_time.hour,
        minute=scheduled_time.minute,
        id="lifeos-morning-brief",
        replace_existing=True,
    )
    return scheduler
