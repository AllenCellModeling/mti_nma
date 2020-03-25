#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from datastep import Step, log_run_params

# Run matplotlib in the background
matplotlib.use('Agg')

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class CompareNucCell(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
        filepath_columns: List[str] = [
            "compare_hist_FilePath",
            "compare_kde_FilePath",
        ]
    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            config=config,
            filepath_columns=filepath_columns,
        )

    @log_run_params
    def run(self, nma_nuc_df, nma_cell_df, **kwargs):
        """
        Run a pure function.

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Parameters
        ----------
        nma_nuc_df: dataframe
            dataframe containing results from running NMA step on nucleus
            See the construction of the manifest in nma.py for details
        nma_cell_df: dataframe
            dataframe containing results from running NMA step on cell
            See the construction of the manifest in nma.py for details
        Returns
        -------
        result: Any
            A pickable object or value that is the result of any processing you do.
        """

        if nma_cell_df or nma_nuc_df is None:
            path = Path(__file__).parent.parent.parent.parent / "local_staging"
            w_nuc = np.load(
                path / "nma_nuc/nma_data/eigvals_Nuc.npy")
            w_cell = np.load(
                path / "nma_cell/nma_data/eigvals_Cell.npy")
        else:
            w_nuc = np.load(nma_nuc_df["w_FilePath"])
            w_cell = np.load(nma_nuc_df["w_FilePath"])

        plt.clf()

        # set binning
        w_all = np.concatenate((w_nuc, w_cell))
        minval = min(w_all) - 0.5
        maxval = max(w_all) + 0.5
        if len(w_nuc) < 20:
            N = int(max(w_all) + 2)
        else:
            N = 30
        bins = np.linspace(minval, maxval, N)

        sb.distplot(w_nuc, kde=False, bins=bins, label="Nuc", norm_hist=True)
        sb.distplot(w_cell, kde=False, bins=bins, label="Cell", norm_hist=True)
        plt.xlabel("Eigenvalues (w2*m/k)")
        plt.ylabel("Counts")
        plt.legend()

        plt.savefig(self.step_local_staging_dir / "compare_fig_hist")

        plt.clf()

        sb.distplot(w_nuc, kde=True, hist=False, bins=bins, label="Nuc")
        sb.distplot(w_cell, kde=True, hist=False, bins=bins, label="Cell")
        plt.xlabel("Eigenvalues (w2*m/k)")
        plt.ylabel("Counts")
        plt.legend()

        plt.savefig(self.step_local_staging_dir / "compare_fig_kde")

        self.manifest = pd.DataFrame({
            "compare_hist_FilePath": (
                self.step_local_staging_dir / "compare_fig_hist.pdf"
            ),
            "compare_kde_FilePath": (
                self.step_local_staging_dir / "compare_fig_kde.pdf"
            ),
        }, index=[0])

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / f"manifest.csv", index=False
        )
        return self.manifest
