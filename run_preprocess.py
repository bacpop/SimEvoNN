from nn_model.preprocess_data import preprocess
import pandas as pd


def save_data(df:pd.DataFrame, path:str):
    df.to_csv(path, index=False)


def main(args):
    df = pd.read_csv(args.input_csv)
    df = preprocess(df, args.scaler_type, args.corr_threshold, args.feature_na_percentage)
    save_data(df, args.output_csv)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess simulation results data for training')
    parser.add_argument('--input_csv', type=str, help='Path to simulation results', required=True)
    parser.add_argument('--output_csv', type=str, help='Path to save preprocessed data', required=True)
    parser.add_argument('--scaler_type', type=str, help='Type of scaler to use from "max_abs", "min_max", "standard", "robust"', required=False, default="max_abs"),# options=["max_abs", "min_max", "standard", "robust"])
    parser.add_argument('--corr_threshold', type=float, help='Correlation threshold for feature selection', required=False, default=0.95)
    parser.add_argument('--feature_na_percentage', type=float, help='Percentage of missing values allowed in a feature', required=False, default=0.5)
    args = parser.parse_args()

    main(args)


