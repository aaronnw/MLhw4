import pandas as pd
datafile = "hw4data.csv"


def load_data():
    df = pd.read_csv(datafile, header=0)
    print(df)

if __name__ == "__main__":
    load_data()
