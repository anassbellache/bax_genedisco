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

import numpy as np
import torch
from torch import Tensor
from botorch.sampling import IIDNormalSampler
from slingpy import AbstractDataSource

from .gp_model import BotorchCompatibleGP
from .utils import dict_to_namespace, jaccard_similarity


class Algorithm(ABC):
    def __init__(self, params):
        self.exe_path = None
        self.params = dict_to_namespace(params)

    def initialize(self):
        self.exe_path = Namespace(x=[], y=[])

    def get_next_x(self):
        next_x = np.random.uniform() if len(self.exe_path.x) < 10 else None
        return next_x

    def take_step(self, f):
        x = self.get_next_x()
        if x is not None:
            y = f(x)
            self.exe_path.append(x)
            self.exe_path.append(y)

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
        """Return distance function for pairs of outputs."""

        # Default dist_fn casts outputs to arrays and returns Euclidean distance
        def dist_fn(a, b):
            a_arr = np.array(a)
            b_arr = np.array(b)
            return np.linalg.norm(a_arr - b_arr)

        return dist_fn


class FixedPathAlgorithm(Algorithm):
    """
    Algorithm with a fixed execution path input sequence, specified by x_path parameter.
    """

    def __init__(self, params):
        super().__init__(params)
        self.params.name = getattr(params, "[[][]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]name",
                                   "FixedPathAlgorithm")
        self.params.x_path = getattr(params, "x_path", [])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        x_path = self.params.x_path
        next_x = x_path[len_path] if len_path < len(x_path) else None
        return next_x

    def get_output(self):
        """Return output based on self.exe_path."""
        # Default behavior: return execution path
        return self.exe_path


class TopK(FixedPathAlgorithm):
    """
    Algorithm that scans over a set of points, and as output returns the K points with
    highest value.
    """

    def __init__(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().__init__(params)

        self.params.name = getattr(params, "name", "TopK")
        self.params.opt_mode = getattr(params, "opt_mode", "max")
        self.params.k = getattr(params, "k", 3)
        self.params.dist_str = getattr(params, "dist_str", "norm")

    def get_exe_path_topk_idx(self):
        """Return the index of the optimal point in execution path."""
        if self.params.opt_mode == "min":
            topk_idx = np.argsort(self.exe_path.y)[:self.params.k]
        elif self.params.opt_mode == "max":
            rev_exe_path_y = -np.array(self.exe_path.y)
            topk_idx = np.argsort(rev_exe_path_y)[:self.params.k]

        return topk_idx

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        topk_idx = self.get_exe_path_topk_idx()

        exe_path_crop = Namespace()
        exe_path_crop.x = [self.exe_path.x[idx] for idx in topk_idx]
        exe_path_crop.y = [self.exe_path.y[idx] for idx in topk_idx]

        return exe_path_crop

    def get_output(self):
        """Return output based on self.exe_path."""
        topk_idx = self.get_exe_path_topk_idx()

        out_ns = Namespace()
        out_ns.x = [self.exe_path.x[idx] for idx in topk_idx]
        out_ns.y = [self.exe_path.y[idx] for idx in topk_idx]

        return out_ns

    def get_output_dist_fn(self):
        """Return distance function for pairs of outputs."""
        if self.params.dist_str == "norm":
            dist_fn = self.output_dist_fn_norm
        elif self.params.dist_str == "jaccard":
            dist_fn = self.output_dist_fn_jaccard

        return dist_fn

    def output_dist_fn_norm(self, a, b):
        """Output dist_fn based on concatenated vector norm."""
        a_list = []
        list(map(a_list.extend, a.x))
        a_list.extend(a.y)
        a_arr = np.array(a_list)

        b_list = []
        list(map(b_list.extend, b.x))
        b_list.extend(b.y)
        b_arr = np.array(b_list)

        return np.linalg.norm(a_arr - b_arr)

    def output_dist_fn_jaccard(self, a, b):
        """Output dist_fn based on Jaccard similarity."""
        a_x_tup = [tuple(x) for x in a.x]
        b_x_tup = [tuple(x) for x in b.x]
        jac_sim = jaccard_similarity(a_x_tup, b_x_tup)
        dist = 1 - jac_sim
        return dist


class SubsetSelect(Algorithm):
    def __init__(self, X: AbstractDataSource, device, n_samples=1000, k=3):
        """
            Initialize the SubsetSelect algorithm.

            :param X: A finite sample set
            :param n_samples: Number of Monte Carlo samples.
            :param k: Number of subset points to select.
        """
        super().__init__(params={})
        self.k = k
        self.X = X
        self.device = device
        self.selected_subset = []
        self.mc_samples = n_samples
        self.exe_path = dict_to_namespace({"x": [], "y": []})
        self.input_dim = torch.tensor(X.get_data(), dtype=torch.float32).shape[-1]

    def initialize(self):
        self.exe_path = dict_to_namespace({"x": [], "y": []})

    def monte_carlo_expectation(self, S: AbstractDataSource, f: BotorchCompatibleGP):
        """
        Estimate the expectation of the maximum value using Monte Carlo sampling with BoTorch.

        :param S: Set of candidate points as an AbstractDataSource.
        :param f: GP model derived from BaseGPModel
        :return: Expected max value across the sampled functions for each candidate point.
        """
        print("monte_carlo expectation")

        # Convert candidate set to tensor
        if isinstance(S, torch.Tensor):
            S_tensor = S.to(self.device)
        else:
            S_tensor = torch.tensor(S.get_data(), dtype=torch.float32, device=self.device)

        # Set up the sampler
        mc_sampler = IIDNormalSampler(sample_shape=torch.Size([self.mc_samples]))

        # Sample from the posterior
        with torch.no_grad():
            posterior = f.posterior(S_tensor)
            samples = mc_sampler(posterior)
            samples = samples.squeeze(-1)  # Reshape as needed

        # Calculate the max value
        max_values = samples.max(dim=0).values
        expected_max = max_values.mean(dim=0).tolist()

        print("monte carlo done")
        return expected_max

    def handle_first_selection(self, f: BotorchCompatibleGP):
        # Calculate Monte Carlo expected values for entire dataset self.X
        scores = self.monte_carlo_expectation(self.X, f)
        TE = torch.tensor(self.X.get_data(), dtype=torch.float32, device=self.device).squeeze(0)

        # Select the element with the highest score
        scores_tensor = torch.tensor(scores, device=self.device)
        first_selection = TE[torch.argmax(scores_tensor).item()]

        return first_selection

    def select_next(self, f: BotorchCompatibleGP):
        """
        Select the next element for the subset based on Monte Carlo estimated scores.

        :return: Point from X that is estimated to maximize the expectation.
        """
        print("select next")
        if len(self.selected_subset) < 1:
            # Handle the first selection
            next_selection = self.handle_first_selection(f)
            return next_selection

        TE = torch.tensor(self.X.get_data(), dtype=torch.float32, device=self.device)

        # Convert the selected subset to a tensor once
        selected_tensor = torch.stack(
            [torch.tensor(selected, device=self.device).float() for selected in self.selected_subset]
        )
        selected_tensor = selected_tensor.to(self.device)

        # Create the mask based on selected subset
        mask = torch.tensor(
            [not any(torch.allclose(x.float(), y) for y in selected_tensor) for x in TE],
            dtype=torch.bool,
            device=self.device
        )

        # Extract candidates using the mask
        candidates_from_mask = TE[mask]
        selected_tensor = selected_tensor.unsqueeze(0)

        # Concatenate selected with candidates
        concatenated_candidates = torch.cat([selected_tensor, candidates_from_mask], dim=1).to(self.device)

        # Calculate the scores using Monte Carlo
        scores = self.monte_carlo_expectation(concatenated_candidates, f)

        # Select the next candidate based on highest score
        scores_tensor = torch.tensor(scores, device=self.device)
        max_index = torch.argmax(scores_tensor).item()
        next_selection = concatenated_candidates[0, max_index]

        return next_selection

    def take_step(self, f: BotorchCompatibleGP):
        """
        Perform one step of the subset selection process.

        Selects the next best point and appends it to the selected subset.
        Also, appends the corresponding output from the base GP to the execution path.

        :param f: Not used in this method but retained for compatibility.
        :return: Next selected point or None if selection is complete.
        """
        print("take step")
        next_x = self.select_next(f)
        next_x = next_x.to(self.device)

        if len(self.selected_subset) < self.k:
            self.selected_subset.append(next_x)

            # Get the prediction using the predict method
            y_pred = f.model(next_x)
            # If y_pred is a list (mean, std, margins), then take the mean.
            y = y_pred[0] if isinstance(y_pred, list) else y_pred
            y_pred_sample = y.sample()
            self.update_exe_paths(next_x, y_pred_sample)
            return next_x
        else:
            return None

    def update_exe_paths(self, x: Tensor, y: Tensor):
        """
        Update the execution paths with the newly selected point and its value.
        """
        self.exe_path.x.append(x.detach().cpu().numpy())
        self.exe_path.y.append(y.detach().cpu().numpy())

    def get_output(self):
        """
        Get the currently selected subset of points.
        :return: List of points selected so far.
        """
        return self.selected_subset

    def get_exe_paths(self, f: BotorchCompatibleGP):
        """
        Get the execution paths for x values and their corresponding model predictions.
        """
        self.initialize()
        with torch.no_grad():
            x = self.take_step(f)
            while x is not None:
                x = self.take_step(f)
        return self.exe_path