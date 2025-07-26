# Builds corpus of news articles on the same story in sequential order

from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)
logger.info("Pulling Dataset")
ds = load_dataset("LogeshChandran/newsroom", split="train")
print(ds.description)
print(ds.keys())


