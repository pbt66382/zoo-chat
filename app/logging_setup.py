"""
配置应用级日志：轮转写入 logs/app.log，便于排查 500 等问题。
"""
import logging
import logging.handlers

from config.settings import get_settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    level_name = settings.log_level
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        settings.app_log_path,
        maxBytes=settings.log_file_max_bytes,
        backupCount=settings.log_file_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    # 终端仍可见 ERROR 及以上，便于本地开发
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.setLevel(logging.ERROR)
    root.addHandler(stream)

    _configured = True
