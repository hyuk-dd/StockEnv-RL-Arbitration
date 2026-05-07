import pathlib

#import finrl

import pandas as pd
import datetime
import os
#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10


#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#DATASET_DIR = PACKAGE_ROOT / "data"

# data
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
# TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"
#
# now = datetime.datetime.now()
# TRAINED_MODEL_DIR = f"trained_models/{now}"
# os.makedirs(TRAINED_MODEL_DIR)
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
#
# TESTING_DATA_FILE = "test.csv"

import os

# ================================
# Directory settings
# ================================
TRAINED_MODEL_DIR = "./trained_models"
RESULTS_DIR = "./results"
DATASET_DIR = "./datasets"

os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ================================
# Technical indicator list
# ================================
# FinRL: ["macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma"
]

# ================================
# Environment hyperparams
# ================================
INITIAL_AMOUNT = 1_000_000
TRANSACTION_FEE_PERCENT = 0.001
HMAX = 100
REWARD_SCALING = 1e-4

