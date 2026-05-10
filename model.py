# ================================
# Imports
# ================================
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
# Prefer Gymnasium if available
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from prc import *
import config

# FinRL env
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

import numpy as np
from collections import deque
import csv

class BayesRelEstimator:
    def __init__(self, memory_size=30, categories=3, threshold=0.002, target_category=2):
        self.categories = categories
        self.threshold = threshold
        self.target_category = target_category
        self.pe_records = deque(maxlen=memory_size)
        self.pe_counts = np.zeros(categories, dtype=np.int64)

    def _cat(self, pe):
        if pe < -self.threshold: return 1
        elif pe > +self.threshold: return 2
        else: return 0

    def add(self, pe):
        if len(self.pe_records) == self.pe_records.maxlen:
            self.pe_counts[self.pe_records[0]] -= 1
        c = self._cat(pe)
        self.pe_records.append(c)
        self.pe_counts[c] += 1

    def _post_mean(self, c):
        K = self.categories
        N = len(self.pe_records)
        return (1 + self.pe_counts[c]) / (K + N)

    def _post_var(self, c):
        K = self.categories
        N = len(self.pe_records)
        num = (1 + self.pe_counts[c]) * (K + N - (1 + self.pe_counts[c]))
        den = (K + N) ** 2 * (K + N + 1)
        return num / (den + 1e-12)

    def reliability(self):
        chis = []
        for c in range(self.categories):
            m = self._post_mean(c)
            v = self._post_var(c)
            chis.append(m / (v + 1e-12))
        rel = chis[self.target_category] / (np.sum(chis) + 1e-12)
        return float(np.clip(rel, 0.0, 1.0))

def fix_data(df: pd.DataFrame, all_tics: list) -> pd.DataFrame:
    df = df.rename(columns={"datadate": "date"})
    df = (
        df.groupby("date")
          .apply(lambda x: (
              x.set_index("tic")
               .reindex(all_tics)
               .ffill()
               .fillna(0)
               .reset_index()
               .assign(date=x['date'].iloc[0])
          ))
          .reset_index(drop=True)
    )

    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    day_index = pd.factorize(df["date"])[0]
    df.index = day_index

    return df

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"datadate": "date"})
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    day_index = pd.factorize(df["date"])[0]
    df.index = day_index
    return df

def ensure_datadate_int(df: pd.DataFrame, col: str = "datadate") -> pd.DataFrame:

    if col not in df.columns:
        df = df.reset_index()

    if np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y%m%d").astype(int)
    else:

        try:

            s = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
            if s.notna().any():

                use = s.fillna(pd.to_datetime(df[col], errors="coerce"))
                df[col] = use.dt.strftime("%Y%m%d").astype(int)
            else:

                df[col] = df[col].astype(str).str.replace("-", "", regex=False).astype(int)
        except Exception:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y%m%d").astype(int)

    df = df.reset_index(drop=True)
    df = df.sort_values([col, "tic"], ignore_index=True)
    return df

def data_split(df: pd.DataFrame, start: int, end: int, col: str = "datadate") -> pd.DataFrame:

    df = ensure_datadate_int(df, col)
    out = df[(df[col] >= start) & (df[col] < end)].copy()
    out.index = out[col].factorize()[0]
    return out

def safe_quantile(x: np.ndarray, q: float, fallback: float = None) -> float:

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return fallback if fallback is not None else 0.0
    try:
        return float(np.nanquantile(x, q))
    except Exception:
        return fallback if fallback is not None else float(np.nanmedian(x))

def make_env_kwargs(df_slice: pd.DataFrame,
                    turbulence_threshold: float | None = None,
                    iteration: int | None = None,
                    initial_amount: float = 1_000_000_0,
                    buy_cost_pct: float = 1e-3,
                    sell_cost_pct: float = 1e-3,
                    hmax: int = 100) -> dict:

    #df_slice = ensure_datadate_int(df_slice)
    stock_dim = int(df_slice["tic"].nunique())
    tech_list = list(getattr(config, "TECHNICAL_INDICATORS_LIST", []))
    state_space = 1 + 2 * stock_dim + len(tech_list) * stock_dim

    kwargs = {
        "df": df_slice,
        "stock_dim": stock_dim,
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [buy_cost_pct] * stock_dim,
        "sell_cost_pct": [sell_cost_pct]* stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_list,
    }
    if turbulence_threshold is not None:
        kwargs["turbulence_threshold"] = float(turbulence_threshold)
    if iteration is not None:
        kwargs["iteration"] = int(iteration)
    return kwargs

def train_A2C(env_train, model_name, timesteps=25_000):
    t0 = time.time()
    model = A2C("MlpPolicy", env_train, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(f"/home/hail/Desktop/model/us/{model_name}")
    print("Training time (A2C):", round((time.time() - t0) / 60, 2), "min")
    return model

def train_PPO(env_train, model_name, timesteps=50_000):
    t0 = time.time()
    model = PPO("MlpPolicy", env_train, ent_coef=0.005, batch_size=64, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(f"/home/hail/Desktop/model/us/{model_name}")
    print("Training time (PPO):", round((time.time() - t0) / 60, 2), "min")
    return model

def train_DDPG(env_train, model_name, timesteps=10_000):
    n_actions = env_train.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                sigma=0.5 * np.ones(n_actions))
    t0 = time.time()
    model = DDPG("MlpPolicy", env_train, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(f"/home/hail/Desktop/model/us/{model_name}")
    print("Training time (DDPG):", round((time.time() - t0) / 60, 2), "min")
    return model

# ================================
# Validation / Prediction
# ================================

def DRL_validation(model, test_data: pd.DataFrame, test_env, test_obs) -> None:

    n_steps = len(test_data["date"].unique())
    for _ in range(n_steps):
        action, _ = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)

def get_validation_sharpe(iteration: int) -> float:

    path = f"results/account_value_validation_JP_{iteration}_{iteration}.csv"
    df_total_value = pd.read_csv(path, index_col=None)
    df_total_value = df_total_value[["account_value"]].rename(columns={"account_value": "account_value_train"})
    df_total_value["daily_return"] = df_total_value["account_value_train"].pct_change(1)
    return (4 ** 0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std()

def get_validation_sharpe_(iteration: int, name) -> float:

    path = f"results/account_value_validation_{name}_{iteration}.csv"
    df_total_value = pd.read_csv(path, index_col=None)
    df_total_value = df_total_value[["account_value"]].rename(columns={"account_value": "account_value_train"})
    df_total_value["daily_return"] = df_total_value["account_value_train"].pct_change(1)
    df_total_value['account_value_train'] = df_total_value['account_value_train'] - 1000000
    return (4 ** 0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std(), df_total_value['account_value_train'].sum() / 1000000

def DRL_prediction(df: pd.DataFrame,
                   model,
                   name: str,
                   last_state: list,
                   iter_num: int,
                   unique_trade_date: np.ndarray,
                   rebalance_window: int,
                   turbulence_threshold: float,
                   initial: bool):

    trade_data = data_split(
        df,
        start=int(unique_trade_date[iter_num - rebalance_window]),
        end=int(unique_trade_date[iter_num])
    )
    all_tics = trade_data['tic'].unique()
    trade_data = fix_data(trade_data, all_tics)
    env_kwargs_trade = make_env_kwargs(trade_data,
                                       turbulence_threshold=turbulence_threshold,
                                       iteration=iter_num)

    if "previous_state" in StockTradingEnv.__init__.__code__.co_varnames:
        env_kwargs_trade["previous_state"] = last_state
    if "initial" in StockTradingEnv.__init__.__code__.co_varnames:
        env_kwargs_trade["initial"] = bool(initial)

    env_trade = DummyVecEnv([lambda: StockTradingEnv(**env_kwargs_trade)])
    obs_trade = env_trade.reset()

    n_steps = len(trade_data["date"].unique())

    for t in range(n_steps):
        action, _ = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)

        if t == n_steps - 2:
            try:
                last_state = env_trade.render()
            except Exception:
                pass

    env_trade.close()

    try:
        pd.DataFrame({"last_state": last_state}).to_csv(
            f"results/last_state_{name}_{iter_num}.csv", index=False
        )
    except Exception:
        pass



    return last_state


def compute_sharpe(returns: List[float], annualize: bool = True, eps: float = 1e-12) -> float:
    """Compute Sharpe ratio from a list of per-step returns."""
    r = np.asarray(returns, dtype=float).flatten()
    if r.size < 2:
        return 0.0
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma < eps:
        return 0.0
    sharpe = mu / (sigma + eps)
    return float(sharpe * np.sqrt(252)) if annualize else float(sharpe)


def _build_val_env(validation_df: pd.DataFrame, i: int, model_name: str) -> DummyVecEnv:
    """Fresh validation env per model for fair rollouts."""
    return DummyVecEnv([lambda: StockTradingEnv(
        **make_env_kwargs(validation_df, iteration=i),
        mode="validation",
        model_name=f"{model_name}_arb_{i}"
    )])


def validate_and_update_reliability(
        model,
        validation_df: pd.DataFrame,
        i: int,
        model_name: str,
        estimator: "BayesRelEstimator") :
    env = _build_val_env(validation_df, i, model_name)
    n_steps = len(validation_df["date"].unique())
    obs = env.reset()
    step_returns: List[float] = []

    for _ in range(n_steps):
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        r = float(rewards)
        r = r / 10000000
        step_returns.append(r)
        estimator.add(r)

    rel = estimator.reliability()
    record = estimator.pe_records

    env.close()
    sharpe = compute_sharpe(step_returns, annualize=True)
    return sharpe, rel, record


def run_arbit_strategy(df: pd.DataFrame,
                       unique_trade_date: np.ndarray,
                       rebalance_window: int,
                       validation_window: int,
                       i: int = 0) -> None:
    print("============Start Ensemble Strategy============")

    df = ensure_datadate_int(df)

    # Sharpe ratio records
    ppo_sharpe_list, a2c_sharpe_list, ddpg_sharpe_list = [], [], []

    baseline_start = 20160101
    baseline_end = int(unique_trade_date[i - rebalance_window - validation_window])
    insample = df[(df.datadate < baseline_end) & (df.datadate >= baseline_start)]

    insample = insample.drop_duplicates(subset=["datadate"])
    insample_thr = safe_quantile(insample.get("turbulence", pd.Series(dtype=float)).values, 0.90, fallback=0.0)

    rel_trackers: Dict[str, BayesRelEstimator] = {
        "A2C": BayesRelEstimator(memory_size=30, categories=3, threshold=0.05, target_category=2),
        "PPO": BayesRelEstimator(memory_size=30, categories=3, threshold=0.05, target_category=2),
        "DDPG": BayesRelEstimator(memory_size=30, categories=3, threshold=0.05, target_category=2),
    }

    #t0 = time.time()

    print("============================================")
    #initial = (i - rebalance_window - validation_window == 0)
    end_date = int(unique_trade_date[i - rebalance_window - validation_window])

    try:
        end_idx = df.index[df["datadate"] == end_date].to_list()[-1]
    except IndexError:

        prev_dates = df.loc[df["datadate"] <= end_date, "datadate"].unique()

        if len(prev_dates) == 0:
            raise RuntimeError("No valid end date for historical turbulence window.")

        end_idx = df.index[df["datadate"] == prev_dates[-1]].to_list()[-1]

    start_idx = max(0, end_idx - validation_window * 30 + 1)
    hist_tb = df.iloc[start_idx:end_idx + 1, :].drop_duplicates(subset=["datadate"])
    hist_tb_mean = float(np.nanmean(hist_tb.get("turbulence", pd.Series(dtype=float)).values))

    if hist_tb_mean > insample_thr:
        turbulence_threshold = insample_thr
    else:

        turbulence_threshold = safe_quantile(insample.get("turbulence", pd.Series(dtype=float)).values,
                                             0.99, fallback=insample_thr)
    print("turbulence_threshold:", turbulence_threshold)

    train = data_split(df,
                       start=20160101,
                       end=int(unique_trade_date[i - rebalance_window - validation_window]))

    all_tics = train['tic'].unique()
    train = fix_data(train, all_tics)
    env_train = DummyVecEnv([lambda: StockTradingEnv(**make_env_kwargs(train))])

    validation = data_split(df,
                            start=int(unique_trade_date[i - rebalance_window - validation_window]),
                            end=unique_trade_date[i - rebalance_window])
    all_tics_ = validation['tic'].unique()
    validation = fix_data(validation, all_tics_)

    # ===== Training & Validation =====
    print("======A2C Training========")
    model_a2c = train_A2C(env_train, model_name=f"A2C_arb_{i}", timesteps=30_000)
    sharpe_a2c, rel_a2c, r1 = validate_and_update_reliability(model_a2c, validation, i, "A2C", rel_trackers["A2C"])
    # sharpe_a2c, ret = get_validation_sharpe_(i, f"A2C_arb_{i}")
    # print("A2C Sharpe Ratio:", sharpe_a2c)

    print("======PPO Training========")
    model_ppo = train_PPO(env_train, model_name=f"PPO_arb_{i}", timesteps=100_000)
    sharpe_ppo, rel_ppo, r2 = validate_and_update_reliability(model_ppo, validation, i, "PPO", rel_trackers["PPO"])
    # sharpe_ppo, ret = get_validation_sharpe_(i,f"PPO_arb_{i}")
    # print("PPO Sharpe Ratio:", sharpe_ppo)

    print("======DDPG Training========")
    model_ddpg = train_DDPG(env_train, model_name=f"DDPG_arb_{i}", timesteps=10_000)
    sharpe_ddpg, rel_ddpg, r3 = validate_and_update_reliability(model_ddpg, validation, i, "DDPG", rel_trackers["DDPG"])
    # sharpe_ddpg, ret = get_validation_sharpe_(i,f"DDPG_arb_{i}")
    # print("DDPG Sharpe Ratio:", sharpe_ddpg)

    # Record Sharpe ratios
    ppo_sharpe_list.append(sharpe_ppo)
    a2c_sharpe_list.append(sharpe_a2c)
    ddpg_sharpe_list.append(sharpe_ddpg)

    # Update reliability trackers
    # rel_trackers["A2C"].add(ret)
    # rel_trackers["PPO"].add(ret)
    # rel_trackers["DDPG"].add(ret)

    # Compute reliability for each model
    reco = {"A2C": r1, 'PPO': r2, 'DDPG': r3}
    reliabilities = {"A2C": rel_a2c, 'PPO': rel_ppo, 'DDPG': rel_ddpg}
    # print("Model reliabilities:", reliabilities)

    df = pd.DataFrame([reliabilities])
    df2 = pd.DataFrame([reco])

    df.to_csv(f"/home/hail/Desktop/model/us/rel/reliabilities_{i}.csv", index=False)
    df2.to_csv(f"/home/hail/Desktop/model/us/rel/records_{i}.csv", index=False)



def run_ensemble_strategy(df: pd.DataFrame,
                          unique_trade_date: np.ndarray,
                          rebalance_window: int,
                          validation_window: int,
                          i: int) -> None:

    print("============Start Ensemble Strategy============")
    df = ensure_datadate_int(df)

    last_state_ensemble: list = []
    model_use = []
    ppo_sharpe_list, a2c_sharpe_list, ddpg_sharpe_list = [], [], []
    shar_ = []

    baseline_start = 20160101
    baseline_end = int(unique_trade_date[i-rebalance_window-validation_window])
    insample = df[(df.datadate < baseline_end) & (df.datadate >= baseline_start)]

    #insample = df[(df.datadate < 20181231) & (df.datadate >= 20160000)]
    insample = insample.drop_duplicates(subset=["datadate"])
    insample_thr = safe_quantile(insample.get("turbulence", pd.Series(dtype=float)).values, 0.90, fallback=0.0)

    t0 = time.time()


    print("============================================")
    initial = (i - rebalance_window - validation_window == 0)
    end_date = int(unique_trade_date[i - rebalance_window - validation_window])

    try:
        end_idx = df.index[df["datadate"] == end_date].to_list()[-1]
    except IndexError:

        prev_dates = df.loc[df["datadate"] <= end_date, "datadate"].unique()

        if len(prev_dates) == 0:
            raise RuntimeError("No valid end date for historical turbulence window.")

        end_idx = df.index[df["datadate"] == prev_dates[-1]].to_list()[-1]

    start_idx = max(0, end_idx - validation_window * 30 + 1)
    hist_tb = df.iloc[start_idx:end_idx + 1, :].drop_duplicates(subset=["datadate"])
    hist_tb_mean = float(np.nanmean(hist_tb.get("turbulence", pd.Series(dtype=float)).values))


    if hist_tb_mean > insample_thr:
        turbulence_threshold = insample_thr
    else:

        turbulence_threshold = safe_quantile(insample.get("turbulence", pd.Series(dtype=float)).values,
                                             0.99, fallback=insample_thr)
    print("turbulence_threshold:", turbulence_threshold)

    # ----------------
    # Train env
    # ----------------
    train = data_split(df,
                       start=20160101,
                       end=int(unique_trade_date[i - rebalance_window - validation_window]))
    #train = prepare(train)
    all_tics = train['tic'].unique()
    train = fix_data(train, all_tics)
    env_train = DummyVecEnv([lambda: StockTradingEnv(**make_env_kwargs(train))])

    # ----------------
    # Validation env
    # ----------------
    validation = data_split(df,
                            start=int(unique_trade_date[i - rebalance_window - validation_window]),
                            end=unique_trade_date[i - rebalance_window])
    #validation = prepare(validation)
    all_tics_ = validation['tic'].unique()
    validation = fix_data(validation, all_tics_)
    env_val = DummyVecEnv([lambda: StockTradingEnv(
        **make_env_kwargs(validation,
                          turbulence_threshold=turbulence_threshold,
                          iteration=i),
        mode="validation",
        model_name=f"JP_{i}"
    )])
    obs_val = env_val.reset()

    # ===== Training & Validation =====
    print("======A2C Training========")
    model_a2c = train_A2C(env_train, model_name=f"A2C_{i}", timesteps=30_000)
    DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
    sharpe_a2c = get_validation_sharpe(i); print("A2C Sharpe Ratio:", sharpe_a2c)

    print("======PPO Training========")
    model_ppo = train_PPO(env_train, model_name=f"PPO_{i}", timesteps=100_000)
    DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
    sharpe_ppo = get_validation_sharpe(i); print("PPO Sharpe Ratio:", sharpe_ppo)

    print("======DDPG Training========")
    model_ddpg = train_DDPG(env_train, model_name=f"DDPG_{i}", timesteps=10_000)
    DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
    sharpe_ddpg = get_validation_sharpe(i); print("DDPG Sharpe Ratio:", sharpe_ddpg)

    ppo_sharpe_list.append(sharpe_ppo)
    a2c_sharpe_list.append(sharpe_a2c)
    ddpg_sharpe_list.append(sharpe_ddpg)
    shar_.append(sharpe_ppo)
    shar_.append(sharpe_a2c)
    shar_.append(sharpe_ddpg)

    with open(f"/home/hail/FinRL-Trading/old_repo_ensemble_strategy/models_arb/select/output_sharpe{i}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(shar_)

    if sharpe_ppo >= sharpe_a2c and sharpe_ppo >= sharpe_ddpg:
        model_ensemble = model_ppo; model_use.append("PPO")
    elif sharpe_a2c >= sharpe_ppo and sharpe_a2c >= sharpe_ddpg:
        model_ensemble = model_a2c; model_use.append("A2C")
    else:
        model_ensemble = model_ddpg; model_use.append("DDPG")

    with open(f"/home/hail/FinRL-Trading/old_repo_ensemble_strategy/models_arb/select/output{i}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(model_use)

    #     # ===== Trading =====
    # print("======Trading from:", int(unique_trade_date[i - rebalance_window]), "to", int(unique_trade_date[i]))
    # last_state_ensemble = DRL_prediction(df=df,
    #                                      model=model_ensemble,
    #                                      name="ensemble",
    #                                      last_state=last_state_ensemble,
    #                                      iter_num=i,
    #                                      unique_trade_date=unique_trade_date,
    #                                      rebalance_window=rebalance_window,
    #                                      turbulence_threshold=turbulence_threshold,
    #                                      initial=initial)
    #
    # print("Ensemble Strategy took:", round((time.time() - t0) / 60, 2), "min")





