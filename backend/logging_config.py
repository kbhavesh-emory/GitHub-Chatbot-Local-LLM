# backend/logging_config.py
import logging

logger = logging.getLogger("backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
