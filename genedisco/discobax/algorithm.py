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


def compute_expected_max(candidate, S_tensor, f, mc_samples, device, mc_sampler=None):
    assert (
        candidate.dim() == 2
    ), f"Expected candidate of dim 2, got {candidate.dim()} with shape {candidate.shape}"
    assert (
        S_tensor.dim() == 2
    ), f"Expected S_tensor of dim 2, got {S_tensor.dim()} with shape {S_tensor.shape}"

    combined = torch.cat([S_tensor, candidate], dim=0)

    # Create the sampler if not provided
    if mc_sampler is None:
        mc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

    with torch.no_grad():
        posterior = f.posterior(combined)
        samples = mc_sampler(posterior).squeeze(-1)

    # Calculate the max value
    max_values = samples.max(dim=0).values
    expected_max = max_values.mean(dim=0).item()
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
        # Ensure candidates are on the correct device
        candidates = candidates.to(self.device)

        S_tensor = torch.stack(S).to(self.device)

        # Parallel computation of expected maxima
        n_jobs = -1
        expected_maxima = Parallel(n_jobs=n_jobs)(
            delayed(compute_expected_max)(
                candidate.unsqueeze(0), S_tensor, f, self.mc_samples, self.device
            )
            for candidate in candidates
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

        mask = torch.tensor(
            [
                not any(torch.allclose(x, y) for y in self.selected_subset)
                for x in self.X
            ],
            dtype=torch.bool,
        ).to(self.device)

        candidates = self.X[mask]

        scores = self.monte_carlo_expectation(candidates, self.selected_subset, f)
        max_index = torch.argmax(torch.tensor(scores)).item()

        return candidates[max_index]

    def take_step(self, f: BotorchCompatibleGP):
        next_x = self.select_next(f)

        if len(self.selected_subset) < self.k:
            self.selected_subset.append(next_x)
            y_pred = f.predict(next_x)
            y = y_pred[0] if isinstance(y_pred, list) else y_pred
            self.update_exe_paths(next_x, y)

            return next_x
        return None

    def update_exe_paths(self, x: torch.Tensor, y: torch.Tensor):
        self.exe_path.x.append(x.cpu().numpy())
        self.exe_path.y.append(y.cpu().numpy())

    def get_output(self):
        return self.selected_subset

    def get_exe_paths(self, f: BotorchCompatibleGP):
        paths = []

        for _ in tqdm(range(self.num_paths), desc="Sampling paths"):
            self.initialize()
            with torch.no_grad():
                x = self.take_step(f)
                while x is not None:
                    x = self.take_step(f)
            paths.append(copy.deepcopy(self.exe_path))

        return paths
