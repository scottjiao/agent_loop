"""Local tool — 查询系统时间和时区信息。"""

from __future__ import annotations

import datetime

from agent_loop.tools.local import LocalToolProvider

provider = LocalToolProvider(provider_id="system_time")


@provider.tool()
async def get_system_time() -> str:
    """获取当前系统时间、日期和时区信息。"""
    now = datetime.datetime.now()
    tz_aware = datetime.datetime.now(datetime.timezone.utc).astimezone()
    tz_name = tz_aware.tzname() or "Unknown"
    utc_offset = tz_aware.strftime("%z")
    offset_formatted = f"UTC{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else "UTC"

    return (
        f"当前系统时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"时区: {tz_name} ({offset_formatted})\n"
        f"星期: {now.strftime('%A')}"
    )
