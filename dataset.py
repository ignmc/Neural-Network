import pandas as pd


def normalize(x, dl, dh, nl, nh):
    return ((x - dl) * (nh - nl)) / (dh - dl) + nl


def encode_expected(possible_outputs):
    possible_outputs.sort()
    d = {}
    for output in possible_outputs:
        encoded = [1 if output == it_output else 0 for it_output in possible_outputs]
        d[output] = encoded
    return d

def get_processed_data(filename):
    df = pd.read_csv(filename, sep=';')
    df = df.sample(frac=1)
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

    un = labels.unique()
    encoded = encode_expected(un)

    labels = [encoded[label] for label in labels]
    return df.values, labels
