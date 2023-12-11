from typing import List, AnyStr

import numpy as np
import torch
from botorch.sampling import SobolQMCNormalSampler
from slingpy import AbstractDataSource, AbstractBaseModel
from tqdm import tqdm

from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from .algorithm import SubsetSelect


class DiscoBAXAdditive(BaseBatchAcquisitionFunction):
    def __init__(self, device, path_sample_num=5, n_components=None) -> None:
        super(DiscoBAXAdditive, self).__init__()
        self.device = device
        self.path_sample_num = path_sample_num
        self.n_components = n_components

    def perform_pca(self, X, n_components):
        # Centering the data
        X_mean = torch.mean(X, dim=0)
        X_centered = X - X_mean

        # SVD
        U, S, V = torch.svd(X_centered)

        # Select the principal components
        components = V[:, :n_components]

        # Project the data onto principal components
        X_pca = torch.mm(X_centered, components)
        return X_pca

    def __call__(
        self,
        dataset_x: AbstractDataSource,
        batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        last_model: AbstractBaseModel,
    ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        X = torch.tensor(
            np.array(avail_dataset_x.get_data()),
            dtype=torch.float32,
            device=self.device,
        ).squeeze(0)

        self.model = last_model
        # Perform PCA if n_components is specified
        if self.n_components is not None:
            X = self.perform_pca(X, self.n_components)

        self.algo = SubsetSelect(
            avail_dataset_x,
            num_paths=self.path_sample_num,
            device=self.device,
            k=batch_size,
        )
        exe_paths = self.algo.get_exe_paths(self.model)
        all_x = [namespace.x for namespace in exe_paths]
        self.xs_exe = torch.stack(all_x).to(self.device)

        # Compute EIG using the current model
        p = self.model.posterior(X)
        h_current = 0.5 * torch.log(2 * torch.pi * p.variance) + 0.5

        total_eig = torch.zeros(X.shape[0], device=self.device)

        # Loop over each execution path
        for xs in tqdm(self.xs_exe, desc="Calculating EIG over paths"):
            # Construct fantasy models using BoTorch's fantasize method
            sampler = SobolQMCNormalSampler(self.path_sample_num)
            fmodels_path = self.model.fantasize(xs, sampler)

            # For fantasy models of this execution path
            pfs = fmodels_path.posterior(X)
            h_fantasies = 0.5 * torch.log(2 * torch.pi * pfs.variance) + 0.5
            avg_h_fantasy = torch.mean(h_fantasies, dim=0)

            eig_path = h_current - avg_h_fantasy
            total_eig += eig_path

        # Average the EIG over all execution paths
        avg_eig = total_eig / len(self.xs_exe)

        # Select top points based on avg_eig
        _, top_indices = torch.topk(avg_eig, batch_size)
        selected_indices = [available_indices[i.item()] for i in top_indices]

        return selected_indices
