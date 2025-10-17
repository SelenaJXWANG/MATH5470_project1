import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join('data', 'application_train.csv')
OUT_DIR = 'plots'
OUT_FILE = os.path.join(OUT_DIR, 'ext_source3_kde.png')

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Make sure you're running from the repo root.")

    df = pd.read_csv(DATA_PATH)

    col = 'EXT_SOURCE_3'
    if col not in df.columns:
        raise KeyError(f"Column {col} not found in {DATA_PATH}")

    # Drop missing values for plotting
    df_plot = df[[col, 'TARGET']].dropna(subset=[col]).copy()

    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.kdeplot(df_plot.loc[df_plot['TARGET'] == 0, col], label='TARGET == 0', fill=True)
    sns.kdeplot(df_plot.loc[df_plot['TARGET'] == 1, col], label='TARGET == 1', fill=True)

    plt.title('Distribution of EXT_SOURCE_3 by Target Value')
    plt.xlabel('EXT_SOURCE_3')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    # Show the plot interactively instead of saving to a file
    plt.show()

if __name__ == '__main__':
    main()
