from typing import List, AnyStr

import numpy as np
import torch
from slingpy import AbstractDataSource, AbstractBaseModel
from botorch.sampling import IIDNormalSampler

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
        X = torch.tensor(np.array(avail_dataset_x.get_data()), dtype=torch.float32, device=self.device)
        self.model = last_model

        # Get execution paths (assumed unchanged)
        self.algo = SubsetSelect(avail_dataset_x, device=self.device)
        exe_path = self.algo.get_exe_paths(self.model)
        self.xs_exe, self.ys_exe = torch.tensor(np.array(exe_path.x), dtype=torch.float32,
                                                device=self.device), torch.tensor(np.array(exe_path.y),
                                                                                  dtype=torch.float32,
                                                                                  device=self.device)

        # Construct fantasy models using BoTorch's fantasize method
        sampler = IIDNormalSampler(self.xs_exe.shape[0])
        self.fmodels = self.model.fantasize(self.xs_exe, sampler)

        # Compute EIG using both the current model and the fantasy models

        # For current model
        p = self.model.posterior(X)
        h_current = 0.5 * torch.log(2 * torch.pi * p.variance) + 0.5

        # For fantasy models
        pfs = self.fmodels.posterior(X)
        h_fantasies = 0.5 * torch.log(2 * torch.pi * pfs.variance) + 0.5
        avg_h_fantasy = torch.mean(h_fantasies, dim=-2)

        eig = h_current - avg_h_fantasy

        # Select top points based on EIG
        _, top_indices = torch.topk(eig, batch_size)
        flattened_top_indices = top_indices.flatten()
        selected_indices = [available_indices[i.item()] for i in flattened_top_indices]

        return selected_indices
