import logging
from datetime import timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.recommender import train_and_refresh_models
from app.core.settings import settings

_SCHEDULER: Optional[AsyncIOScheduler] = None
logger = logging.getLogger(__name__)
_DAY_MAP = {
    "monday": "mon",
    "tuesday": "tue",
    "wednesday": "wed",
    "thursday": "thu",
    "friday": "fri",
    "saturday": "sat",
    "sunday": "sun",
}


def _get_scheduler() -> AsyncIOScheduler:
    global _SCHEDULER
    if _SCHEDULER is None:
        _SCHEDULER = AsyncIOScheduler(timezone=timezone.utc)
    return _SCHEDULER

def _normalize_day(spec: str) -> str:
    if not spec:
        return "sun"
    values = []
    for part in (segment.strip().lower() for segment in spec.split(",") if segment.strip()):
        if part in _DAY_MAP:
            values.append(_DAY_MAP[part])
        elif part in _DAY_MAP.values():
            values.append(part)
        elif part.isdigit() and 0 <= int(part) <= 6:
            values.append(part)
        else:
            raise ValueError(f"Invalid weekday name '{part}'")
    if not values:
        raise ValueError("No valid weekday provided for retraining schedule")
    return ",".join(values)


def start_scheduler() -> None:
    scheduler = _get_scheduler()
    try:
        day_spec = _normalize_day(settings.RETRAIN_DAY)
    except ValueError as exc:
        logger.warning("Invalid RETRAIN_DAY '%s', defaulting to Sunday: %s", settings.RETRAIN_DAY, exc)
        day_spec = "sun"
    scheduler.add_job(
        train_and_refresh_models,
        trigger="cron",
        day_of_week=day_spec,
        hour=settings.RETRAIN_HOUR_UTC,
        minute=0,
        id="weekly-retraining",
        replace_existing=True,
    )
    if not scheduler.running:
        scheduler.start()


def stop_scheduler() -> None:
    scheduler = _get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=False)


def trigger_now() -> None:
    train_and_refresh_models(manual_trigger=True)
