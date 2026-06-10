import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm
from loguru import logger

logger.add('log/running.log', rotation='5MB')
