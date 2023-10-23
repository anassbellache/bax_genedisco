from typing import List, AnyStr

import numpy as np
import torch
from slingpy import AbstractDataSource, AbstractBaseModel

from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from .algorithm import SubsetSelect


class DiscoBAXAdditive(BaseBatchAcquisitionFunction):
    r"""Single outcome expected information gain`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EIG = ExpectedInformationGain(model, algo, algo_params)
        >>> eig = EIG(test_X)
    """

    def __init__(
            self,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            maximize: If True, consider the problem a maximization problem.
        """
        super(DiscoBAXAdditive).__init__()
        self.device = torch.device("cpu")

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel,
                 ) -> List:
        """
        Nominate experiments for the next learning round using Expected Information Gain.

        Args:
            dataset_x: The dataset containing all training samples.
            batch_size: Size of the batch to acquire.
            available_indices: The list of the indices (names) of the samples not
                chosen in the previous rounds.
            last_selected_indices: The set of indices selected in the previous
                cycle.
            last_model: The prediction model trained by labeled samples chosen so far.

        Returns:
            A list of indices (names) of the samples chosen for the next round.
        """

        # Subset the available data points
        avail_dataset_x = dataset_x.subset(available_indices)
        X = torch.tensor(avail_dataset_x.get_data(), dtype=torch.float32, device=self.device)
        self.model = last_model
        self.algo = SubsetSelect(avail_dataset_x, device=self.device)

        # Get execution paths using SubsetSelect
        exe_path = self.algo.get_exe_paths(self.model)
        self.xs_exe, self.ys_exe = np.array(exe_path.x), np.array(exe_path.y)
        self.xs_exe = torch.tensor(self.xs_exe, dtype=torch.float32, device=self.device)
        self.ys_exe = torch.tensor(self.ys_exe, dtype=torch.float32, device=self.device)


        # Construct a batch of fantasy models
        self.fmodels = self.model.condition_on_observations(self.xs_exe, self.ys_exe)

        # Calculate the variance of the posterior for current data
        p = self.model.posterior(X)
        var_p = p.variance.reshape(p.variance.shape[:-1])

        # Calculate the variance of the fantasy posteriors
        pfs = self.fmodels.posterior(X)
        var_pfs = pfs.variance.reshape(pfs.variance.shape[:-1])

        # Calculate Shannon entropy for current and fantasy posteriors
        h_current = 0.5 * torch.log(2 * torch.pi * var_p) + 0.5
        h_fantasies = 0.5 * torch.log(2 * torch.pi * var_pfs) + 0.5

        # Compute EIG
        avg_h_fantasy = torch.mean(h_fantasies, dim=-2)
        eig = h_current - avg_h_fantasy

        # Get indices of points with the highest EIG
        _, top_indices = torch.topk(eig, batch_size)
        flattened_top_indices = top_indices.flatten()
        selected_indices = [available_indices[i.item()] for i in flattened_top_indices]

        return selected_indices
