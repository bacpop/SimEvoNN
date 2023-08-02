import numpy as np
import pandas as pd


def get_high_corr_features(df:pd.DataFrame, column_ids_dict:dict, corr_threshold:float=0.95):
    ### Deal with high correlations
    pd_features = df.iloc[:, :-2]
    # Compute the correlation matrix
    corr_matrix = pd_features.corr().abs()

    # Get the upper triangular matrix
    upper_triangular = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with high correlation
    high_corr_features = {column for column in upper_triangular.columns if
                          any(upper_triangular[column] > corr_threshold)}

    return {column_ids_dict[c_id] for c_id in high_corr_features}


def get_no_var_features(df:pd.DataFrame, column_ids_dict:dict):
    cols_with_0_var = np.isclose(df.var(), 0)
    cols_with_0_var_set = {column_ids_dict[col] for col in cols_with_0_var.nonzero()[0]}

    return cols_with_0_var_set
    #return df.columns[df.nunique() == 1]


def drop_nas_and_dubs(df:pd.DataFrame, feature_na_percentage = 0.5):
    ### We do not want to use these simulation parameters
    df.drop(
        columns=[
            "max_mutations",
            "n_generations"
        ], inplace=True)

    df = df.drop_duplicates()

    ### Drop columns are mostly nans
    column_na_freq_fltr = df.isnull().sum() / max(df.count()) > feature_na_percentage
    df.drop(columns=df.loc[:, column_na_freq_fltr].columns, inplace=True)

    ### Drop column if all 0.0s
    cols_dropped = df.any(axis=0)
    df = df.loc[:, cols_dropped]

    ### Drop row if all 0.0s
    rows_dropped = df.iloc[:, :-2].any(axis=1)
    df = df.loc[rows_dropped, :]

    ### Drop rows all containing na
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=1, how="all")

    ### Fill or drop na
    ### Drop all rows containing na for now

    pd_naless = df.dropna(axis=0, how="any")

    return pd_naless


def normalise(df:pd.DataFrame, scaler_type:str="max_abs"):
    if scaler_type == "max_abs":
        from sklearn.preprocessing import MaxAbsScaler
        my_scaler = MaxAbsScaler().fit(df)
    elif scaler_type == "min_max":
        from sklearn.preprocessing import MinMaxScaler
        my_scaler = MinMaxScaler().fit(df)
    elif scaler_type == "standard":
        from sklearn.preprocessing import StandardScaler
        my_scaler = StandardScaler().fit(df)
    elif scaler_type == "robust":
        from sklearn.preprocessing import RobustScaler
        my_scaler = RobustScaler().fit(df)
    else:
        raise ValueError("Scaler type not supported")

    _data = my_scaler.transform(df)
    return pd.DataFrame(_data), my_scaler


def preprocess(df:pd.DataFrame, scaler_type:str="max_abs", corr_threshold:float=0.95, feature_na_percentage:float=0.5):
    df = drop_nas_and_dubs(df, feature_na_percentage)
    df_normalised, my_scaler = normalise(df, scaler_type)
    df_columns_id_dict = {i: c for i, c in enumerate(df.columns)}
    high_corr_features = get_high_corr_features(df_normalised, df_columns_id_dict, corr_threshold)
    no_var_feature = get_no_var_features(df_normalised, df_columns_id_dict)
    df.drop(columns=high_corr_features.union(no_var_feature), inplace=True)
    return df


