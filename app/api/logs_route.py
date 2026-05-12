"""
开发/排障用：返回 app.log 末尾若干行（可通过环境变量关闭）。
"""
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query

from config.settings import get_settings

router = APIRouter(prefix="/api/logs", tags=["logs"])


def _tail_file(path: Path, max_lines: int) -> List[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    out = [ln.rstrip("\n\r") for ln in lines[-max_lines:]]
    return out


@router.get("/tail")
def tail_app_log(
    lines: int = Query(200, ge=1, le=2000, description="返回日志文件最后 N 行"),
):
    """
    返回 ``logs/app.log`` 末尾行文本列表。

    默认在 ``DEBUG=true`` 时开启；生产环境请设置 ``LOG_HTTP_TAIL=false`` 关闭。
    """
    settings = get_settings()
    if not settings.log_http_tail:
        raise HTTPException(
            status_code=404,
            detail="日志查询接口已关闭。请在 .env 中设置 LOG_HTTP_TAIL=true（仅建议在受信环境使用）。",
        )

    raw = _tail_file(settings.app_log_path, lines)
    return {
        "path": str(settings.app_log_path),
        "line_count": len(raw),
        "lines": raw,
    }
