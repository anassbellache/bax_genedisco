from typing import List, AnyStr

import numpy as np
import torch
from slingpy import AbstractDataSource, AbstractBaseModel
from botorch.sampling import IIDNormalSampler
from tqdm import tqdm
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
            monte_carlo_num=10
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            maximize: If True, consider the problem a maximization problem.
        """
        super(DiscoBAXAdditive).__init__()
        self.device = torch.device("cpu")
        self.monte_carlo_num = monte_carlo_num

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
        self.algo = SubsetSelect(avail_dataset_x, num_paths=self.monte_carlo_num, device=self.device)
        exe_paths = self.algo.get_exe_paths(self.model)
        if isinstance(exe_paths, list):
            # handle the list case
            self.xs_exe = torch.tensor(np.array([np.array(exe_path.x) for exe_path in exe_paths]), dtype=torch.float32, device=self.device)
        else:
            # handle the object case
            self.xs_exe = torch.tensor(np.array(exe_paths.x), dtype=torch.float32, device=self.device)


        # Compute EIG using both the current model and the fantasy models
        # For current models
        p = self.model.posterior(X)
        h_current = 0.5 * torch.log(2 * torch.pi * p.variance) + 0.5

        total_eig = torch.zeros(X.shape[0], device=self.device)

        # Loop over each execution path
        for i in tqdm(range(self.xs_exe.shape[0])):
            # Construct fantasy models using BoTorch's fantasize method for this execution path
            sampler = IIDNormalSampler(self.monte_carlo_num)
            fmodels_path = self.model.fantasize(self.xs_exe[i], sampler)

            # For fantasy models of this execution path
            pfs = fmodels_path.posterior(X)
            h_fantasies = 0.5 * torch.log(2 * torch.pi * pfs.variance) + 0.5
            avg_h_fantasy = torch.mean(h_fantasies, dim=-2)

            eig_path = h_current - avg_h_fantasy
            total_eig += eig_path

        # Average the EIG over all execution paths
        avg_eig = total_eig / self.xs_exe.shape[0]

        # Select top points based on avg_eig
        _, top_indices = torch.topk(avg_eig.squeeze(), batch_size, dim=-1)
        flattened_top_indices = top_indices.flatten()
        selected_indices = [available_indices[i.item()] for i in flattened_top_indices]

        return selected_indices

