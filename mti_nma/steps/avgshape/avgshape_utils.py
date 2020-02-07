import matplotlib.pyplot as plt
from pathlib import Path


def run_shcoeffs_analysis(df, savedir):

    list_of_scatter_plots = [
        ('shcoeffs_L0M0C', 'shcoeffs_L2M0C'),
        ('shcoeffs_L0M0C', 'shcoeffs_L2M2C'),
        ('shcoeffs_L0M0C', 'shcoeffs_L2M1S'),
        ('shcoeffs_L0M0C', 'shcoeffs_L2M1C'),
    ]

    for id_plot, (varx, vary) in enumerate(list_of_scatter_plots):

        fs = 18
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(df[varx], df[vary], 'o')
        ax.set_xlabel(varx, fontsize=fs)
        ax.set_ylabel(vary, fontsize=fs)
        plt.tight_layout()
        fig.savefig(
            str(savedir / Path(f'scatter-{id_plot}.svg'))
        )
        plt.close(fig)

    list_of_bar_plots = [
        'shcoeffs_chi2',
        'shcoeffs_L0M0C',
    ]

    for id_plot, var in enumerate(list_of_bar_plots):

        fs = 18
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.bar(df.index, df[var])
        ax.set_xlabel('CellId', fontsize=10)
        ax.set_ylabel(var, fontsize=fs)
        ax.tick_params('x', labelrotation=90)
        plt.tight_layout()
        fig.savefig(
            str(savedir / Path(f'bar-{id_plot}.svg'))
        )
        plt.close(fig)
