import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import warnings


## Dataset class: for loading IMC datasets
class SCDataset(Dataset):
    """Pytorch holder for numpy data
    """

    def __init__(self, expr_input, m_proteins: List[str], design: np.array) -> None:
        self._m_proteins = m_proteins
        if isinstance(expr_input, pd.DataFrame):
            self._exprs, self._exprs_X = self._process_df_input(expr_input)
            self._expr_proteins = list(expr_input.columns)
            self._core_names = list(expr_input.index)
        elif isinstance(expr_input, tuple):
            self._expr_proteins = expr_input[1]
            self._core_names = expr_input[2]
            self._exprs, self._exprs_X = self._process_np_input(expr_input[0])
        self.design = self._fix_design(design)

        if self._exprs.shape[0] <= 0:
            raise NotClassifiableError(
                "Classification failed. There "
                + "should be at least one row of data to be classified."
            )

    def _process_df_input(self, df_input):
        try:
            Y_np = df_input[self._m_proteins].to_numpy()
        except (KeyError):
            raise NotClassifiableError(
                "Classification failed. There's no "
                + "overlap between marked proteins and expression proteins for "
                + "the classification of cell type/state."
            )
        X = StandardScaler().fit_transform(Y_np)
        return torch.from_numpy(Y_np), torch.from_numpy(X)

    def _process_np_input(self, np_input):
        ind = [self._expr_proteins.index(name) for name in self._m_proteins
            if name in self._expr_proteins]
        if len(ind) <= 0:
            raise NotClassifiableError(
                "Classification failed. There's no "
                + "overlap between marked proteins and expression proteins for "
                + "the classification of cell type/state."
            )
        if len(ind) < len(self._m_proteins):
            warnings.warn("Classified proteins are less than marked proteins.")
        Y_np = []
        for cell in np_input:
            temp = [cell[i] for i in ind]
            Y_np.append(np.array(temp))
        Y_np = np.concatenate([Y_np], axis = 0)
        X = StandardScaler().fit_transform(Y_np)
        return torch.from_numpy(Y_np), torch.from_numpy(X)

    def __len__(self) -> int:
        return self._exprs.shape[0]

    def __getitem__(self, idx):
        return self._exprs[idx, :], self._exprs_X[idx, :], self.design[idx, :]

    def _fix_design(self, design: np.array) -> torch.tensor:
        d = None
        if design is None:
            d = torch.ones((self._exprs.shape[0], 1)).double()
        else:
            d = torch.from_numpy(design).double()

        if d.shape[0] != self._exprs.shape[0]:
            raise NotClassifiableError(
                "Number of rows of design matrix "
                + "must equal number of rows of expression data"
            )
        return d
        
    def get_exprs(self):
        return self._exprs

    def get_exprs_X(self):
        return self._exprs_X

    def get_mu(self):
        return self._exprs.mean(0)

    def get_sigma(self):
        return self._exprs.std(0)

    def get_class_amount(self):
        return self._exprs.shape[1]

    def get_protein_amount(self):
        return len(self._m_proteins)

    def get_proteins(self):
        return self._m_proteins

    def get_cells(self):
        return self._core_names


class NotClassifiableError(RuntimeError):
    pass
