import pandas as pd


if __name__ == "__main__":
    train_df = pd.read_fwf("data/artificial_train.data")
    train_labels = pd.read_fwf("data/artificial_train.labels")
    test_df = pd.read_fwf("data/artificial_valid.data")
