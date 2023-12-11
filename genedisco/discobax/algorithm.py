# TODO:
#  Takes a function and runs a set of predefined computations on it
#  Algorithm must be compatible with model class
#  Decoupled from the function f
#  Make it iterative
#  Understads that it will recieve an AbstractBaseModel
#  Has a distance function
#  Works on embedding spaces of genedisco
#  Works in batch mode

import copy
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List

import numpy as np
import torch
from botorch.sampling import SobolQMCNormalSampler
from joblib import Parallel, delayed
from slingpy import AbstractDataSource
from tqdm import tqdm

from .gp_model import BotorchCompatibleGP
from .utils import dict_to_namespace, jaccard_similarity


class Algorithm(ABC):
    def __init__(self, params):
        self.exe_path = Namespace(x=[], y=[])
        self.params = dict_to_namespace(params)

    def initialize(self):
        self.exe_path = Namespace(x=[], y=[])

    def get_next_x(self):
        return np.random.uniform() if len(self.exe_path.x) < 10 else None

    def take_step(self, f):
        x = self.get_next_x()
        if x is not None:
            y = f(x)
            self.exe_path.x.append(x)
            self.exe_path.y.append(y)
        return x

    @abstractmethod
    def get_output(self):
        pass

    def run_algorithm_on_f(self, f):
        self.initialize()
        x = self.take_step(f)
        while x is not None:
            x = self.take_step(f)
        return self.exe_path, self.get_output()

    def get_copy(self):
        return copy.deepcopy(self)

    def get_exe_path_crop(self):
        return self.exe_path

    def get_output_dist_fn(self):
        def dist_fn(a, b):
            a_arr = np.array(a)
            b_arr = np.array(b)
            return np.linalg.norm(a_arr - b_arr)

        return dist_fn


class FixedPathAlgorithm(Algorithm):
    def __init__(self, params):
        super().__init__(params)
        self.params.name = getattr(params, "name", "FixedPathAlgorithm")
        self.params.x_path = getattr(params, "x_path", [])

    def get_next_x(self):
        len_path = len(self.exe_path.x)
        return (
            self.params.x_path[len_path] if len_path < len(self.params.x_path) else None
        )

    def get_output(self):
        return self.exe_path


class TopK(FixedPathAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.params.name = getattr(params, "name", "TopK")
        self.params.opt_mode = getattr(params, "opt_mode", "max")
        self.params.k = getattr(params, "k", 3)
        self.params.dist_str = getattr(params, "dist_str", "norm")

    def get_exe_path_topk_idx(self):
        return (
            np.argsort(self.exe_path.y)[: self.params.k]
            if self.params.opt_mode == "min"
            else np.argsort(self.exe_path.y)[-self.params.k :]
        )

    def get_exe_path_crop(self):
        topk_idx = self.get_exe_path_topk_idx()
        return Namespace(
            x=[self.exe_path.x[idx] for idx in topk_idx],
            y=[self.exe_path.y[idx] for idx in topk_idx],
        )

    def get_output(self):
        topk_idx = self.get_exe_path_topk_idx()
        return Namespace(
            x=[self.exe_path.x[idx] for idx in topk_idx],
            y=[self.exe_path.y[idx] for idx in topk_idx],
        )

    def get_output_dist_fn(self):
        return (
            self.output_dist_fn_jaccard
            if self.params.dist_str == "jaccard"
            else self.output_dist_fn_norm
        )

    def output_dist_fn_norm(self, a, b):
        a_arr = np.concatenate((a.x, a.y))
        b_arr = np.concatenate((b.x, b.y))
        return np.linalg.norm(a_arr - b_arr)

    @staticmethod
    def output_dist_fn_jaccard(a, b):
        return 1 - jaccard_similarity(a.x, b.x)


def compute_expected_max(candidates, S_tensor, f, mc_samples, device, mc_sampler=None):
    # Handling a potentially 3-dimensional S_tensor
    if S_tensor.dim() == 3:
        S_tensor = S_tensor.squeeze(0)
    assert (
        S_tensor.dim() == 2
    ), f"Expected S_tensor of dim 2, got {S_tensor.dim()} with shape {S_tensor.shape}"

    assert (
        candidates.dim() == 3
    ), f"Expected candidates of dim 3, got {candidates.dim()} with shape {candidates.shape}"

    # Expand S_tensor to match the batch size of candidates
    S_tensor = S_tensor.unsqueeze(0).expand(candidates.size(0), -1, -1)
    combined = torch.cat([S_tensor, candidates], dim=1).to(device)

    if mc_sampler is None:
        mc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

    with torch.no_grad():
        posterior = f.posterior(combined)
        samples = mc_sampler(posterior).squeeze(-1)

    # Calculate the max value
    max_values = samples.max(dim=1).values  # Max over each MC sample
    expected_max = max_values.mean(dim=1)  # Mean over all candidates
    return expected_max


class SubsetSelect(Algorithm):
    def __init__(self, X: AbstractDataSource, num_paths, device, n_samples=500, k=3):
        super().__init__(params={})
        self.k = k
        self.X = torch.tensor(np.array(X.get_data()), dtype=torch.float32).to(device)
        self.device = device
        self.selected_subset = []
        self.mc_samples = n_samples
        self.num_paths = num_paths
        self.input_dim = self.X.shape[-1]
        self.exe_path = Namespace(x=[], y=[])

    def initialize(self):
        self.exe_path = Namespace(x=[], y=[])
        self.selected_subset = []

    def monte_carlo_expectation(
        self, candidates: torch.Tensor, S: List[torch.Tensor], f: BotorchCompatibleGP
    ):
        candidates = candidates.to(self.device)
        S_tensor = torch.stack(S).to(self.device)

        # Compute expected maxima for all candidates in parallel
        expected_maxima = compute_expected_max(
            candidates, S_tensor, f, self.mc_samples, self.device
        )

        return expected_maxima

    def handle_first_selection(self, f: BotorchCompatibleGP):
        with torch.no_grad():
            posterior = f.posterior(self.X)
            mean = posterior.mean.squeeze()

        # If f has a noise model, add the expected noise to the mean.
        if hasattr(f, "noise_gp") and f.noise_gp is not None:
            noise = f.noise_gp(self.X).mean.squeeze()
            fb_out_expected = mean + noise
        else:
            fb_out_expected = mean

        # Select the element with the highest expected fb_out value
        first_selection_index = torch.argmax(fb_out_expected).item()
        first_selection = self.X[first_selection_index]

        return first_selection

    def select_next(self, f: BotorchCompatibleGP):
        if len(self.selected_subset) == 0:
            return self.handle_first_selection(f)

        # Convert the list of tensors to a single tensor
        if self.selected_subset:
            subset_tensor = torch.stack(self.selected_subset)

            # Ensure tensors are 2D
            X_flat = self.X.view(self.X.size(0), -1)
            subset_flat = subset_tensor.view(subset_tensor.size(0), -1)

            # Calculate distances between each pair of points in X and selected_subset
            distances = torch.cdist(X_flat, subset_flat)

            # Create a mask where True indicates a row in X with no close match in selected_subset
            threshold = 0.01  # Define your closeness threshold
            mask = torch.all(distances > threshold, dim=1)
        else:
            mask = torch.ones(self.X.size(0), dtype=torch.bool)

        candidates = self.X[mask]

        # Reshape candidates to be 3D if necessary
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0)  # Add a batch dimension

        # Reshape candidates if necessary
        if candidates.dim() == 4:
            # Assuming the tensor should be collapsed into 3 dimensions
            # Adjust the following line according to the correct reshaping needed
            candidates = candidates.squeeze(1)

        if candidates.dim() != 3:
            raise ValueError(f"Unexpected shape of candidates: {candidates.shape}")

        scores = self.monte_carlo_expectation(candidates, self.selected_subset, f)
        max_index = torch.argmax(torch.tensor(scores)).item()

        return candidates[max_index]

    def take_step(self, f: BotorchCompatibleGP):
        next_x = self.select_next(f)

        if next_x is None:
            # Handle the case when no next selection is possible
            print("No next selection is possible.")
            return None

        if len(self.selected_subset) < self.k:
            self.selected_subset.append(next_x)
            y_pred = f.predict(
                next_x.unsqueeze(0)
            )  # Ensure next_x has correct dimensions
            y = y_pred[0] if isinstance(y_pred, list) else y_pred
            self.update_exe_paths(next_x, y)

            return next_x
        return None

    def update_exe_paths(self, x: torch.Tensor, y):
        self.exe_path.x.append(x.cpu().numpy())  # Assuming x is always a torch.Tensor
        if isinstance(y, torch.Tensor):
            self.exe_path.y.append(
                y.cpu().numpy()
            )  # Convert PyTorch tensor to NumPy array
        else:
            self.exe_path.y.append(y)  # y is already a NumPy array

    def get_output(self):
        return self.selected_subset

    def get_exe_paths(self, f: BotorchCompatibleGP):
        paths = []
        f = f.to(self.device)

        for _ in tqdm(range(self.num_paths), desc="Sampling paths"):
            self.initialize()
            with torch.no_grad():
                x = self.take_step(f)
                while x is not None:
                    x = self.take_step(f)
            paths.append(copy.deepcopy(self.exe_path))

        return paths
