import logging
import os
from logging.handlers import TimedRotatingFileHandler

# 创建日志目录
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "logs.log")

# 使用具名 logger，避免污染根 logger
logger = logging.getLogger("aide")  # 或你的项目名
logger.setLevel(logging.INFO)

# 防止重复添加 handler
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 按天滚动日志，保留30天的日志文件
    fh = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",  # 每天午夜滚动
        interval=1,  # 每天一次
        backupCount=30,  # 保留30天的日志
        encoding="utf-8",
    )
    fh.suffix = "%Y-%m-%d"  # 日志文件后缀格式
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def get_logger() -> logging.Logger:
    """获取配置好的 logger 实例"""
    return logger


__all__ = ["get_logger", "logger"]
