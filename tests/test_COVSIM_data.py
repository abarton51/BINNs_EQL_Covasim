import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_COVSIM_data():

    file_path = '../Data/COVSIM/'
    data = pd.read_excel(file_path + 'NSF COVSIM Output.xlsx')

    #
    skipped_len = 140
    truncate_len = 100
    data = data.iloc[skipped_len:skipped_len+truncate_len,:].copy()
    col_names = list(data.columns)
    n = len(col_names)
    t = np.arange(data.shape[0])
    fig = plt.figure(figsize=(18, 9))
    for i in range(1, n + 1):
        ax = fig.add_subplot(2, 4, i)
        ax.plot(t, data.iloc[:, i - 1], '.k-', label='COVSIM Data')
        if i > 4:
            ax.set_xlabel("Time (Days)")
        if i % 4 == 1:
            ax.set_ylabel("Count")
        ax.set_title(col_names[i - 1])
        if i == 1:
            ax.legend(loc="best")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(file_path + 'COVSIM_data' + '.png', dpi=300)
