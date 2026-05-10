# preprocessor
from prc import *
# config
from config import *
# model
from model import *
import os


def run_model(mode=1, j=0) -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "/home/hail/Desktop/model/us/en/u_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
        data = data.reset_index()
        data = data.reset_index(drop=True)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    unique_trade_date = data[(data.datadate >= 20190101) & (data.datadate <= 20250631)].datadate.unique()
    print(unique_trade_date)

    rebalance_window = 63
    validation_window = 63

    print(j)

    ## Ensemble Strategy
    if mode == 1:
        run_ensemble_strategy(df=data,
                              unique_trade_date=unique_trade_date,
                              rebalance_window=rebalance_window,
                              validation_window=validation_window, i=j)
    elif mode == 2:
        run_arbit_strategy(df=data,
                           unique_trade_date=unique_trade_date,
                           rebalance_window=rebalance_window,
                           validation_window=validation_window,
                           i=j)

    # _logger.info(f"saving model version: {_version}")


if __name__ == "__main__":
    l = [126, 189, 252, 315, 378, 441, 504, 567, 630, 693, 756, 819, 882, 945, 1008, 1071, 1134, 1197, 1260, 1323, 1386,
         1449, 1512, 1575]
    for ii in l:
        run_model(mode=1, j=ii)

