import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_data(filename):
    data = pd.read_csv(filename)
    df_norm = data.copy()
    df_norm['Time'] = StandardScaler().fit_transform(df_norm['Time'].values.reshape(-1, 1))
    df_norm['Amount'] = StandardScaler().fit_transform(df_norm['Amount'].values.reshape(-1, 1))
    train_x = df_norm[df_norm.Class == 0]
    train_x = train_x.drop(['Class'], axis=1)
    test_y = df_norm['Class']
    test_x = df_norm.drop(['Class'], axis=1)
    return train_x.values, test_x.values, test_y
