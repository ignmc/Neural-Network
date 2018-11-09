import pandas as pd


def normalize(x, dl, dh, nl, nh):
    return ((x - dl) * (nh - nl)) / (dh - dl) + nl


def get_processed_data(filename):
    df = pd.read_csv(filename, sep=';')
    labels = df['quality']
    df = df.drop('quality', axis=1)

    # normalize
    for column_name in df:
        dl = min(df[column_name])
        dh = max(df[column_name])
        nl = 0
        nh = 1
        for i in range(len(df[column_name])):
            x = df[column_name][i]
            df.at[i, column_name] = normalize(x, dl, dh, nl, nh)

    df = df.sample(frac=1)
    n_rows = df.shape[0]
    train = df.iloc[:n_rows//2, :]
    query = df.iloc[n_rows//2:, :]

    return train, query
