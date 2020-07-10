"""
Cell State Model
"""
from typing import Tuple, List, Dict, Union
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from .abstract import AstirModel
from astir.data import SCDataset
from .cellstate_recognet import StateRecognitionNet
from tqdm import trange
from torch.autograd import Variable
from torch.utils.data import DataLoader
import h5py
from collections import OrderedDict


class CellStateModel(AstirModel):
    """Class to perform statistical inference to on the activation
        of states (pathways) across cells

    :param df_gex: the input gene expression dataframe
    :param marker_dict: the gene marker dictionary
    :param random_seed: seed number to reproduce results, defaults to 1234
    :param dtype: torch datatype to use in the model
    """

    def __init__(
        self,
        dset: SCDataset = None,
        const: int = 2,
        dropout_rate: float = 0,
        batch_norm: bool = False,
        random_seed: int = 42,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(dset, random_seed, dtype)

        # Setting random seeds
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._optimizer = None
        self._losses = torch.empty(0, dtype=self._dtype)
        if self._dset is not None:
            self._param_init(const, dropout_rate, batch_norm)

        # Convergence flag
        self._is_converged = False

    def _param_init(self, const, dropout_rate, batch_norm) -> None:
        """ Initializes sets of parameters
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        N = len(self._dset)
        C = self._dset.get_n_classes()
        G = self._dset.get_n_features()

        initializations = {
            "log_sigma": torch.log(self._dset.get_sigma().mean()),
            "mu": torch.reshape(self._dset.get_mu(), (1, -1)),
        }

        # Include beta or not
        d = torch.distributions.Uniform(
            torch.tensor(0.0, dtype=self._dtype), torch.tensor(1.5, dtype=self._dtype)
        )
        initializations["log_w"] = torch.log(d.sample((C, self._dset.get_n_features())))

        self._variables = {
            n: i.to(self._device).detach().clone().requires_grad_()
            for (n, i) in initializations.items()
        }

        self._data = {
            "rho": self._dset.get_marker_mat().T.to(self._device),
        }

        self._recog = StateRecognitionNet(
            C, G, const=const, dropout_rate=dropout_rate, batch_norm=batch_norm
        ).to(device=self._device, dtype=self._dtype)

    def load_hdf5(self, hdf5_name, const, dropout_rate, batch_norm):
        self._assignment = pd.read_hdf(hdf5_name, "cellstate_model/cellstate_assignments")
        with h5py.File(hdf5_name, "r") as f:
            grp = f["cellstate_model"]
            param = grp["parameters"]
            self._variables = {"mu": torch.tensor(np.array(param["mu"])), 
                "log_sigma": torch.tensor(np.array(param["log_sigma"])), 
                "log_w": torch.tensor(np.array(param["log_w"]))}
            self._data = {"rho": torch.tensor(np.array(param["rho"]))}
            self._losses = torch.tensor(np.array(grp["losses"]["losses"]))

            rec = grp["recog_net"]
            hidden1_W = torch.tensor(np.array(rec['linear1.weight']))
            hidden2_W = torch.tensor(np.array(rec['linear2.weight']))
            hidden3_mu_W = torch.tensor(np.array(rec['linear3_mu.weight']))
            hidden3_std_W = torch.tensor(np.array(rec['linear3_std.weight']))
            state_dict = {'linear1.weight': hidden1_W, 
                'linear1.bias': torch.tensor(np.array(rec["linear1.bias"])), 
                'linear2.weight': hidden2_W, 
                'linear2.bias': torch.tensor(np.array(rec["linear2.bias"])), 
                'linear3_mu.weight': hidden3_mu_W, 
                'linear3_mu.bias': torch.tensor(np.array(rec["linear3_mu.bias"])), 
                'linear3_std.weight': hidden3_std_W, 
                'linear3_std.bias': torch.tensor(np.array(rec["linear3_std.bias"]))}
            state_dict = OrderedDict(state_dict)
            self._recog = StateRecognitionNet(hidden3_mu_W.shape[0], hidden1_W.shape[1], 
                const=const, dropout_rate=dropout_rate, batch_norm=batch_norm
            ).to(device=self._device, dtype=self._dtype)
            self._recog.load_state_dict(state_dict)
            self._recog.eval()

    def _loss_fn(
        self,
        mu_z: torch.Tensor,
        std_z: torch.Tensor,
        z_sample: torch.Tensor,
        y_in: torch.Tensor,
    ) -> torch.Tensor:
        """ Returns the calculated loss

        :param mu_z: the predicted mean of z
        :param std_z: the predicted standard deviation of z
        :param z_sample: the sampled z values
        :param y_in: the input data

        :return: the loss
        """
        S = y_in.shape[0]

        # log posterior q(z) approx p(z|y)
        q_z_dist = torch.distributions.Normal(loc=mu_z, scale=torch.exp(std_z))
        log_q_z = q_z_dist.log_prob(z_sample)

        # log likelihood p(y|z)
        rho_w = torch.mul(self._data["rho"], torch.exp(self._variables["log_w"]))
        mean = self._variables["mu"] + torch.matmul(z_sample, rho_w)
        std = torch.exp(self._variables["log_sigma"]).reshape(1, -1)
        p_y_given_z_dist = torch.distributions.Normal(loc=mean, scale=std)
        log_p_y_given_z = p_y_given_z_dist.log_prob(y_in)

        # log prior p(z)
        p_z_dist = torch.distributions.Normal(0, 1)
        log_p_z = p_z_dist.log_prob(z_sample)

        loss = (1 / S) * (
            torch.sum(log_q_z) - torch.sum(log_p_y_given_z) - torch.sum(log_p_z)
        )

        return loss

    def _forward(
        self, Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ One forward pass

        :param Y: dataset to do forward pass on

        :return: mu_z, std_z, z_sample
        """
        mu_z, std_z = self._recog(Y)

        std = torch.exp(std_z)
        eps = torch.randn_like(std)
        z_sample = eps * std + mu_z

        return mu_z, std_z, z_sample

    def fit(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        delta_loss_batch: int = 10,
        msg: str = "",
    ) -> List[float]:
        for l in self.fit_yield_loss(
            max_epochs,
            learning_rate,
            batch_size,
            delta_loss,
            delta_loss_batch,
            msg,
        ):
            pass


    # @profile
    def fit_yield_loss(
        self,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        delta_loss: float = 1e-3,
        delta_loss_batch: int = 10,
        msg: str = "",
    ) -> List[float]:
        """ Runs train loops until the convergence reaches delta_loss for\
            delta_loss_batch sizes or for max_epochs number of times

        :param max_epochs: number of train loop iterations, defaults to 50
        :param learning_rate: the learning rate, defaults to 0.01
        :param batch_size: the batch size, defaults to 128
        :param delta_loss: stops iteration once the loss rate reaches\
            delta_loss, defaults to 0.001
        :param delta_loss_batch: the batch size to consider delta loss,\
            defaults to 10
        :param msg: iterator bar message, defaults to empty string
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        losses = []

        # Returns early if the model has already converged
        if self._is_converged:
            return losses

        if delta_loss_batch >= max_epochs:
            warnings.warn("Delta loss batch size is greater than the number of epochs")

        # Create an optimizer if there is no optimizer
        if self._optimizer is None:
            opt_params = list(self._recog.parameters()) + list(self._variables.values())
            self._optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

        if self._losses.shape[0] >= delta_loss_batch:
            prev_mean = torch.mean(self._losses[-delta_loss_batch:])
        else:
            prev_mean = None

        delta_cond_met = False

        iterator = trange(
            max_epochs,
            desc="training restart" + msg,
            unit="epochs",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        )
        train_iterator = DataLoader(
            self._dset, batch_size=min(batch_size, len(self._dset))
        )
        for ep in iterator:
        # for ep in range(max_epochs):
            for i, (y_in, x_in, _) in enumerate(train_iterator):
                self._optimizer.zero_grad()

                mu_z, std_z, z_samples = self._forward(x_in)

                loss = self._loss_fn(mu_z, std_z, z_samples, x_in)

                loss.backward()

                self._optimizer.step()

            losses.append(loss.cpu().detach().item())

            start_index = ep - delta_loss_batch + 1
            end_index = start_index + delta_loss_batch
            if start_index >= 0:
                curr_mean = sum(losses[start_index:end_index]) / len(
                    losses[start_index:end_index]
                )
            elif self._losses.shape[0] >= -start_index:
                last_ten_losses = torch.cat(
                    (
                        self._losses[start_index:],
                        torch.tensor(losses[:end_index], dtype=torch.float64),
                    )
                )
                curr_mean = torch.mean(last_ten_losses).item()
            else:
                curr_mean = None

            if prev_mean is not None:
                curr_delta_loss = (prev_mean - curr_mean) / prev_mean
                delta_cond_met = 0 <= curr_delta_loss < delta_loss
            iterator.set_postfix_str("current loss: " + str(round(losses[ep], 1)))
            yield round(losses[ep], 1)

            prev_mean = curr_mean
            if delta_cond_met:
                losses = losses[0 : ep + 1]
                self._is_converged = True
                iterator.close()
                break

        if self._losses is None:
            self._losses = torch.tensor(losses, dtype=self._dtype)
        else:
            self._losses = torch.cat(
                (self._losses, torch.tensor(losses, dtype=self._dtype))
            )

        g = self.get_final_mu_z().detach().cpu().numpy()
        self._assignment = pd.DataFrame(g)
        self._assignment.columns = self._dset.get_classes()
        self._assignment.index = self._dset.get_cell_names()

    def get_recognet(self) -> StateRecognitionNet:
        """ Getter for the recognition net

        :return: the trained recognition net
        """
        return self._recog

    def get_final_mu_z(self, new_dset: SCDataset = None) -> torch.Tensor:
        """ Returns the mean of the predicted z values for each core

        :param new_dset: returns the predicted z values of this dataset on
            the existing model. If None, it predicts using the existing dataset

        :return: the mean of the predicted z values for each core
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        if new_dset is None:
            _, x_in, _ = self._dset[:]  # should be the scaled one
        else:
            _, x_in, _ = new_dset[:]
        final_mu_z, _, _ = self._forward(x_in)

        return final_mu_z

    def get_correlations(self) -> np.array:
        state_assignment = self.get_final_mu_z().detach().cpu().numpy()
        y_in = self._dset.get_exprs()

        feature_names = self._dset.get_features()
        state_names = self._dset.get_classes()
        G = self._dset.get_n_features()
        C = self._dset.get_n_classes()
        corr_mat = np.zeros((C, G))
        # Make a matrix of correlations between all states and proteins
        for c, state in enumerate(state_names):
            for g, feature in enumerate(feature_names):
                states = state_assignment[:, c]
                protein = y_in[:, g].cpu()
                corr_mat[c, g] = np.corrcoef(protein, states)[0, 1]

        return corr_mat

    def diagnostics(self) -> pd.DataFrame:
        """ Run diagnostics on cell state assignments

        :return: diagnostics
        """
        if self._dset is None:
            raise Exception("the dataset is not provided")
        feature_names = self._dset.get_features()
        state_names = self._dset.get_classes()

        corr_mat = self.get_correlations()

        # Correlation values of all marker proteins
        marker_mat = self._dset.get_marker_mat().T.cpu().numpy()
        marker_corr = marker_mat * corr_mat
        marker_corr[marker_mat == 0] = np.inf

        # Smallest correlation values for each pathway
        min_marker_corr = np.min(marker_corr, axis=1).reshape(-1, 1)
        min_marker_proteins = np.take(feature_names, np.argmin(marker_corr, axis=1))

        # Correlation values of all non marker proteins
        non_marker_mat = 1 - self._dset.get_marker_mat().T.cpu().numpy()
        non_marker_corr = non_marker_mat * corr_mat
        non_marker_corr[non_marker_mat == 0] = -np.inf

        # Any correlation values where non marker proteins is greater than
        # the smallest correlation values of marker proteins
        bad_corr_marker = np.array(non_marker_corr > min_marker_corr, dtype=np.int32)

        # Problem summary
        indices = np.argwhere(bad_corr_marker > 0)

        col_names = [
            "pathway",
            "protein A",
            "correlation of protein A",
            "protein B",
            "correlation of protein B",
            "note",
        ]

        problems = []
        for index in indices:
            state_index = index[0]
            protein_index = index[1]
            state = state_names[index[0]]
            marker_protein = min_marker_proteins[state_index]
            non_marker_protein = feature_names[protein_index]
            problem = {
                "pathway": state,
                "marker_protein": marker_protein,
                "corr_of_marker_protein": min_marker_corr[state_index][0],
                "non_marker_protein": non_marker_protein,
                "corr_of_non_marker_protein": non_marker_corr[
                    state_index, protein_index
                ],
                "msg": "{} is marker for {} but {} isn't".format(
                    marker_protein, state, non_marker_protein
                ),
            }
            problems.append(problem)

        if len(problems) > 0:
            df_issues = pd.DataFrame(problems)
            df_issues.columns = col_names
        else:
            df_issues = pd.DataFrame(columns=col_names)

        return df_issues


class NotClassifiableError(RuntimeError):
    """ Raised when the input data is not classifiable.
    """
    pass
