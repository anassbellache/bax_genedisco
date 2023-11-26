# TODO:
#  model must be a GP
#  It must have sampling
#  It has to be trainable
#  There are two models, one for experiment, one for noise
#  use gpytorch
#  try to add a Bayseian neural net
#  Ensure compatilibility with slingpy and AbstractBaseModel, AbstractDataSource
#  Update posterior with new data
#  Scales up to large datasets
#  Handles batches
from typing import Any
from typing import (
    AnyStr,
    Type,
    List,
)
from typing import Optional

import botorch
import gpytorch
import numpy as np
import torch
import torch.optim
from botorch.models.model import Model
from botorch.posteriors import Posterior, GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from slingpy import AbstractBaseModel, AbstractDataSource
from torch import Tensor
from torch.nn import Module


class SparseGPModel(ApproximateGP):
    def __init__(self, likelihood, dim_input, device, num_inducing_points=10):
        # Initialize inducing points on the specified device
        inducing_points = torch.randn(num_inducing_points, dim_input, device=device)

        # Setup the variational distribution and ensure it's on the same device
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        ).to(device)

        # Initialize the variational strategy on the specified device
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        # Call superclass's __init__ method with the necessary arguments
        super(SparseGPModel, self).__init__(variational_strategy=variational_strategy)

        # Define mean and covariance functions and ensure they're on the same device
        self.mean_module = gpytorch.means.ZeroMean().to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        ).to(device)

        # Store the likelihood and ensure it's on the same device
        self.likelihood = likelihood.to(device)

        # Store device for further use
        self.device = device

    def forward(self, x):
        # Ensure input x is on the correct device
        x = x.to(self.device)

        # Return the predictive mean and variance for the given inputs x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(
        self, X: Tensor, *args, observation_noise: bool = False, **kwargs: Any
    ) -> GPyTorchPosterior:
        # Ensure input X is on the correct device
        X = X.to(self.device)

        # Get the predictive distribution (prior) for X
        predictive_distribution = self.forward(X)

        # Get the approximate posterior using the variational strategy
        posterior_distribution = self.variational_strategy(X, predictive_distribution)

        # If observation_noise flag is set, add the likelihood noise to the posterior
        if observation_noise:
            posterior_distribution = self.likelihood(
                posterior_distribution, *args, **kwargs
            )

        # Return the posterior as a Botorch GPyTorchPosterior object
        return GPyTorchPosterior(posterior_distribution)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        """
        Condition the model on new observations and return a new model instance.
        """
        # Ensure inputs X and Y are on the correct device
        X, Y = X.to(self.device), Y.to(self.device)

        # Condition the model on new observations
        fantasy_model = self.get_fantasy_model(X, Y, **kwargs)
        return fantasy_model


class LargeFeatureExtractor(torch.nn.Module):
    def __init__(self, data_dim, feature_dim, device):
        super(LargeFeatureExtractor, self).__init__()
        self.device = device
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, feature_dim),
        ).to(self.device)

    def forward(self, x):
        # Ensure input tensor is on the correct device
        x = x.to(self.device)
        # Pass through the layers
        x = self.layers(x)
        return x


class GPLayer(ApproximateGP):
    def __init__(self, num_dim, device):
        self.device = device
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=10
        ).to(self.device)
        inducing_points = torch.zeros(10, num_dim, device=self.device)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPLayer, self).__init__(variational_strategy)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        ).to(self.device)
        self.mean_module = gpytorch.means.ConstantMean().to(self.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, device):
        super(DKLModel, self).__init__()
        self.device = device
        self.feature_extractor = feature_extractor.to(self.device)
        self.gp_layer = GPLayer(num_dim, self.device)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.num_dim = num_dim

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            self.grid_bounds[0], self.grid_bounds[1]
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        # Assume scale_to_bounds is a method that scales features to the bounds required for the GP
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs: Any
    ):
        """
        Condition the model on new observations and return a new model instance.
        """
        # Create a new DKLModel instance with the same parameters as the current model
        conditioned_model = self.__class__(
            feature_extractor=self.feature_extractor,
            num_dim=self.num_dim,
            device=self.device,
        ).to(self.device)

        # Copy over the state dict (parameters) from the current model to the new model
        conditioned_model.load_state_dict(self.state_dict())

        # Use the get_fantasy_model method from the GaussianProcessLayer to condition the GP layer
        conditioned_model.gp_layer = self.gp_layer.get_fantasy_model(X, Y, **kwargs)

        return conditioned_model


class NeuralGPModel(ApproximateGP, botorch.models.model.FantasizeMixin):
    def __init__(
        self,
        likelihood,
        dim_input,
        device,
        feature_selection=100,
        num_inducing_points=10,
    ):
        # Initialize inducing points
        inducing_points = torch.randn(num_inducing_points, dim_input).to(device)

        # Setup the variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        ).to(device)

        # Initialize the variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        # Call superclass's __init__ method with the necessary arguments
        super(NeuralGPModel, self).__init__(variational_strategy=variational_strategy)
        # Define mean and covariance functions
        self.mean_module = gpytorch.means.ZeroMean().to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        ).to(device)
        self.likelihood = likelihood

        # Store device for further use
        self.device = device

        # Neural network feature extractor
        self.feature_extractor = LargeFeatureExtractor(dim_input, feature_selection).to(
            device
        )

    def update_train_data(self, new_x: Tensor, new_y: Tensor) -> None:
        """
        Update the model's training data with new observations.

        Args:
            new_x: New training inputs.
            new_y: New training targets.
        """
        new_x_projected = self.feature_extractor(new_x)

        if self.train_inputs[0] is None:
            updated_train_x = new_x_projected.to(self.device)
            updated_train_y = new_y.to(self.device)
        else:
            train_inputs_projected = self.feature_extractor(self.train_inputs[0]).to(
                self.device
            )

            # Concatenate old and new data
            if train_inputs_projected.dim() == 1:
                updated_train_x = torch.cat(
                    [train_inputs_projected, new_x_projected.squeeze(0)], dim=0
                ).to(self.device)
            else:
                updated_train_x = torch.cat(
                    [train_inputs_projected, new_x_projected], dim=0
                ).to(self.device)

            updated_train_y = torch.cat([self.train_targets, new_y], dim=0).to(
                self.device
            )

        self.set_train_data(updated_train_x, updated_train_y, strict=False)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        """
        Condition the NeuralGP on new observations (X, Y) and return a new NeuralGPModel.
        """
        # Update train data
        self.update_train_data(X, Y)

        # Create a new model instance with updated training data
        new_model = self.__class__(self.data_dim, self.likelihood, self.device)

        # Copy model components
        new_model.likelihood = self.likelihood
        new_model.mean_module = self.mean_module
        new_model.covar_module = self.covar_module
        new_model.feature_extractor = self.feature_extractor

        # Set the training data of the new model
        new_model.set_train_data(self.train_inputs[0], self.train_targets)

        return new_model

    def posterior(
        self, X: Tensor, observation_noise: bool = False, **kwargs: Any
    ) -> Posterior:
        # Process the input through the neural network.
        # Obtain the prior distribution.
        mvn = self(X)

        # If observation noise should be added and the likelihood is GaussianLikelihood
        if observation_noise and isinstance(self.likelihood, GaussianLikelihood):
            noise = self.likelihood.noise
            mvn = MultivariateNormal(
                mvn.mean, mvn.lazy_covariance_matrix.add_diag(noise)
            )

        # Return the botorch wrapper around GPyTorch's posterior.
        return GPyTorchPosterior(mvn)

    def transform_inputs(
        self, X: Tensor, input_transform: Optional[Module] = None
    ) -> Tensor:
        pass

    def forward(self, x):
        # Return the predictive mean and variance for the given inputs x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, **kwargs):
        return self.forward(x)


class SumGPModel(ApproximateGP):
    def __init__(self, neural_gp: DKLModel, noise_gp: SparseGPModel):
        super().__init__(neural_gp.variational_strategy)

        self.neural_gp = neural_gp
        self.noise_gp = noise_gp

        # Extracting the likelihoods directly from the GPs
        self.neural_likelihood = neural_gp.likelihood
        self.noise_likelihood = noise_gp.likelihood

        # Ensure both GPs are on the same device
        self.device = neural_gp.device
        self.neural_gp.to(self.device)
        self.noise_gp.to(self.device)

    def forward(self, x):
        # Ensure input tensor is on the correct device
        x = x.to(self.device)

        neural_out = self.neural_likelihood(self.neural_gp(x))
        noise_out = self.noise_likelihood(self.noise_gp(x))

        mean_x = neural_out.mean + noise_out.mean
        covar_x = neural_out.lazy_covariance_matrix + noise_out.lazy_covariance_matrix

        return MultivariateNormal(mean_x, covar_x)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        # Ensure inputs X and Y are on the correct device
        X, Y = X.to(self.device), Y.to(self.device)

        # Condition both GPs on the new observations
        conditioned_neural_gp = self.neural_gp.gp_layer.get_fantasy_model(
            X, Y, **kwargs
        )
        conditioned_noise_gp = self.noise_gp.get_fantasy_model(X, Y, **kwargs)

        # Create a new SumGPModel instance with the conditioned GPs
        conditioned_model = self.__class__(conditioned_neural_gp, conditioned_noise_gp)
        conditioned_model.to(self.device)

        # Return the new conditioned model
        return conditioned_model


class BotorchCompatibleGP(
    Model, AbstractBaseModel, botorch.models.model.FantasizeMixin
):
    def __init__(self, dim_input, device, batch_size: int = 64):
        super().__init__()

        self.device = device
        self.data_dim = dim_input
        self.batch_size = batch_size

        self.likelihood = GaussianLikelihood().to(device)
        self.noise_likelihood = GaussianLikelihood().to(device)

        self.noise_gp = (
            SparseGPModel(self.noise_likelihood, dim_input, device).to(device).float()
        )
        self.neural_gp = DKLModel(self.likelihood, dim_input, device).to(device).float()
        self.sum_gp = SumGPModel(self.neural_gp, self.noise_gp).to(device)

        self.sum_likelihood = GaussianLikelihood().to(device)

        # Initialization
        self.train_x = None
        self.train_y = None
        self.return_samples = False

    def forward(self, x):
        x = x.to(self.device)
        return self.sum_gp(x)

    def fit(
        self,
        train_x: AbstractDataSource,
        train_y: Optional[AbstractDataSource] = None,
        validation_set_x: Optional[AbstractDataSource] = None,
        validation_set_y: Optional[AbstractDataSource] = None,
    ) -> "AbstractBaseModel":
        if not train_x or not train_y:
            raise ValueError("Both train_x and train_y must be provided")

        # Convert data to tensors
        train_x = torch.tensor(
            np.array(train_x.get_data()), dtype=torch.float32, device=self.device
        ).squeeze(0)
        train_y = torch.tensor(
            np.array(train_y.get_data()), dtype=torch.float32, device=self.device
        ).squeeze(0)

        self.num_samples = train_y.size(0)

        self.sum_gp.train()
        mll_sum = gpytorch.mlls.VariationalELBO(
            self.sum_likelihood, self.sum_gp, self.num_samples
        )
        optimizer_sum = torch.optim.Adam(self.sum_gp.parameters(), lr=0.01)

        for i in range(50):
            for j in range(train_x.shape[0]):
                optimizer_sum.zero_grad()
                output_sum = self.sum_gp(train_x[j])
                loss_sum = -mll_sum(output_sum, train_y[j]).sum()
                loss_sum.backward(retain_graph=True)
                optimizer_sum.step()
            print(f"Epoch {i}, Loss: {loss_sum}")

        return self

    def predict(
        self,
        dataset_x: AbstractDataSource,
        batch_size: int = 256,
        row_names: List[AnyStr] = None,
        return_samples=True,
    ) -> List[np.ndarray]:
        x_tensor = torch.tensor(
            np.array(dataset_x.get_data()), dtype=torch.float32, device=self.device
        )

        self.return_samples = return_samples

        # Set models to evaluation mode
        self.sum_gp.eval()
        self.likelihood.eval()

        # Split the data into batches and predict
        num_samples = x_tensor.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        all_pred_means, all_pred_stds, all_samples = [], [], []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(num_batches):
                batch_x = x_tensor[i * batch_size : (i + 1) * batch_size]
                main_pred = self.likelihood(self.sum_gp(batch_x))

                all_pred_means.append(main_pred.mean.cpu().numpy())
                all_pred_stds.append(main_pred.variance.sqrt().cpu().numpy())

                if return_samples:
                    samples = [main_pred.sample().cpu().numpy() for _ in range(10)]
                    all_samples.append(np.mean(samples, axis=0))

        if not return_samples:
            pred_mean = np.concatenate(all_pred_means, axis=0)
            pred_std = np.concatenate(all_pred_stds, axis=0)
            upper_bound = pred_mean + 1.96 * pred_std
            lower_bound = pred_mean - 1.96 * pred_std
            y_margins = upper_bound - lower_bound
            return [pred_mean, pred_std, y_margins]
        else:
            return np.concatenate(all_samples, axis=0)

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> "BotorchCompatibleGP":
        X, Y = X.to(self.device), Y.to(self.device)
        if self.train_x is not None and self.train_y is not None:
            X = torch.cat([self.train_x, X], dim=0)
            Y = torch.cat([self.train_y, Y], dim=0)

        fantasy_model = BotorchCompatibleGP(self.data_dim, self.device)
        fantasy_model.train_x = X
        fantasy_model.train_y = Y
        return fantasy_model

    def posterior(
        self, X: Tensor, observation_noise: bool = False, **kwargs: Any
    ) -> Posterior:
        X = X.to(self.device)
        mvn = self.sum_gp(X)
        if observation_noise and isinstance(self.likelihood, GaussianLikelihood):
            mvn = MultivariateNormal(
                mvn.mean, mvn.lazy_covariance_matrix.add_diag(self.likelihood.noise)
            )
        return GPyTorchPosterior(mvn)

    @classmethod
    def load(
        cls: Type["BotorchCompatibleGP"], file_path: AnyStr
    ) -> "BotorchCompatibleGP":
        """
        Load the model from the specified file path.

        Parameters:
            - file_path (str): The path from where the model should be loaded.

        Returns:
            - model (BotorchCompatibleGP): The loaded model.
        """
        print("loading model: ")
        # Load the saved state dictionary
        state_dict = torch.load(file_path)

        # Extract data_dim from the saved state
        data_dim = state_dict["data_dim"]
        device = torch.device("cpu")

        # Create a new model instance with the extracted data_dim
        model = cls(data_dim, device)

        # Restore the state of the models and the likelihoods
        model.neural_gp.load_state_dict(state_dict["neural_gp"])
        model.likelihood.load_state_dict(state_dict["likelihood"])
        model.noise_gp.load_state_dict(state_dict["noise_gp"])
        model.noise_likelihood.load_state_dict(state_dict["noise_likelihood"])

        return model

    @staticmethod
    def get_save_file_extension() -> AnyStr:
        return "pt"

    def save(self, file_path: AnyStr):
        """
        Save the model to the specified file path.

        Parameters:
            - file_path (str): The path to where the model should be saved.
        """
        print("savig model: ")
        state_dict = {
            "data_dim": self.data_dim,
            "neural_gp": self.neural_gp.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "noise_gp": self.noise_gp.state_dict(),
            "noise_likelihood": self.noise_likelihood.state_dict(),
        }
        torch.save(state_dict, file_path)
