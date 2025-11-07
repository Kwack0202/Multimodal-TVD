# ======================================================
import FinanceDataReader as fdr
import pandas_datareader.data as pdr

# ======================================================
# TA-Lib
import talib
import argparse

# ======================================================
# basic library
import warnings

import openpyxl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import math
import os
import os.path
import random
import shutil
import glob

import pickle

import re

# ======================================================
# multiprocessing
from multiprocessing import Pool, cpu_count, freeze_support

# ======================================================
# tqdm
from tqdm import tqdm

# ======================================================
# datetime
from datetime import timedelta, datetime

# ======================================================
# plotting library
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

import plotly.graph_objects as go
import plotly.subplots as ms
import plotly.express as px

import mplfinance as fplt
from mplfinance.original_flavor import candlestick2_ohlc, volume_overlay

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.dates import MonthLocator
from PIL import Image
import matplotlib.gridspec as gridspec

import seaborn as sns
import cv2
import csv

# plt.rcParams['figure.dpi'] = 150
from IPython.display import clear_output

# ======================================================
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig