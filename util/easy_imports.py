import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm

logger.add('log/running.log', rotation='5MB')
