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
from typing import Any, Optional
from typing import (
    AnyStr,
    List,
)

import botorch
import gpytorch
import numpy as np
import torch
import torch.optim
from botorch.models.model import Model
from botorch.posteriors import Posterior, GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from slingpy import AbstractBaseModel, AbstractDataSource
from torch import Tensor
from torch.nn import Module


class SparseGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, device):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)

        # Move to the specified device
        self.device = device

        # Define mean and covariance functions
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        # Ensure input tensor is on the correct device
        x = x.to(self.device)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs
    ) -> "NeuralGPModel":
        """
        Condition the model on new observations (X, Y) and return a new GP model.
        """

        X = X.to(self.device)
        Y = Y.to(self.device)
        # Create a fantasy model that includes the new observations
        fantasy_model = self.get_fantasy_model(X, Y, **kwargs)

        return fantasy_model

    def posterior(
        self, X: torch.Tensor, observation_noise: bool = False, **kwargs
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Returns the posterior distribution at given inputs X.
        """
        X = X.to(self.device)
        # Get the predictive distribution for X
        predictive_distribution = self(X)

        if observation_noise:
            # Add observation noise to the predictive distribution if required
            predictive_distribution = self.likelihood(predictive_distribution)

        return predictive_distribution


class LargeFeatureExtractor(torch.nn.Module):
    def __init__(self, data_dim, feature_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, feature_dim),
        )

    def forward(self, x):
        # Pass through the layers
        x = self.layers(x)
        return x


class NeuralGPModel(ExactGP, botorch.models.model.FantasizeMixin):
    def __init__(
        self, train_x, train_y, likelihood, dim_input, device, feature_selection=100
    ):
        super(NeuralGPModel, self).__init__(train_x, train_y, likelihood)

        # Move to the specified device
        self.device = device

        # Define mean and covariance functions
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # Neural network feature extractor
        self.feature_extractor = LargeFeatureExtractor(dim_input, feature_selection).to(
            device
        )

    def forward(self, x):
        # Transform input using the feature extractor
        projected_x = self.feature_extractor(x)

        # Compute mean and covariance using the transformed input
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

    def update_train_data(self, new_x: Tensor, new_y: Tensor) -> None:
        # Project new inputs using the feature extractor
        new_x = new_x.to(self.device)
        new_y = new_y.to(self.device)
        new_x_projected = self.feature_extractor(new_x).to(self.device)

        # Update the training data
        if self.train_inputs[0] is None:
            updated_train_x = new_x_projected
            updated_train_y = new_y
        else:
            # Concatenate old and new data
            updated_train_x = torch.cat([self.train_inputs[0], new_x_projected], dim=0)
            updated_train_y = torch.cat([self.train_targets, new_y], dim=0)

        self.set_train_data(updated_train_x, updated_train_y, strict=False)

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs
    ) -> "NeuralGPModel":
        """
        Condition the model on new observations (X, Y) and return a new GP model.
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        # Create a fantasy model that includes the new observations
        fantasy_model = self.get_fantasy_model(X, Y, **kwargs)

        return fantasy_model

    def posterior(
        self, X: torch.Tensor, observation_noise: bool = False, **kwargs
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Returns the posterior distribution at given inputs X.
        """
        # Get the predictive distribution for X
        X = X.to(self.device)
        predictive_distribution = self(X)

        if observation_noise:
            # Add observation noise to the predictive distribution if required
            predictive_distribution = self.likelihood(predictive_distribution)

        return predictive_distribution

    def transform_inputs(
        self, X: Tensor, input_transform: Optional[Module] = None
    ) -> Tensor:
        pass

    def __call__(self, x, **kwargs):
        x = x.to(self.device)
        return self.forward(x)


class SumGPModel(ExactGP):
    def __init__(
        self,
        neural_gp: NeuralGPModel,
        noise_gp: SparseGPModel,
        train_x,
        train_y,
        device,
    ):
        likelihood = (
            neural_gp.likelihood
            if neural_gp.likelihood == noise_gp.likelihood
            else GaussianLikelihood()
        )
        super(SumGPModel, self).__init__(train_x, train_y, likelihood)

        self.neural_gp = neural_gp
        self.noise_gp = noise_gp
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        # Get outputs from both models
        neural_out = self.neural_gp(x)
        noise_out = self.noise_gp(x)

        # Sum the means and covariances
        mean_x = neural_out.mean + noise_out.mean
        covar_x = neural_out.lazy_covariance_matrix + noise_out.lazy_covariance_matrix

        return MultivariateNormal(mean_x, covar_x)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        """
        Condition the model on new observations and return a new model instance.
        This method might need more sophisticated handling depending on the
        underlying models' capabilities to be conditioned on new data.
        """
        # Update training data for both models
        self.neural_gp.set_train_data(X, Y, strict=False)
        self.noise_gp.set_train_data(X, Y, strict=False)

        # Create a new SumGPModel instance with updated models
        conditioned_model = self.__class__(self.neural_gp, self.noise_gp, X, Y)

        return conditioned_model

    def posterior(
        self, X: Tensor, observation_noise: bool = False, **kwargs: Any
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Returns the posterior distribution of the summed GP model at given inputs X.
        """
        # Ensure the input is on the correct device
        X = X.to(self.device)

        # Get predictive distributions from both models
        neural_pred = self.neural_gp(X)
        noise_pred = self.noise_gp(X)

        if observation_noise:
            # Add observation noise from the likelihoods of each model
            neural_pred = self.neural_likelihood(neural_pred)
            noise_pred = self.noise_likelihood(noise_pred)

        # Sum the means and covariances
        mean_x = neural_pred.mean + noise_pred.mean
        covar_x = neural_pred.lazy_covariance_matrix + noise_pred.lazy_covariance_matrix

        return MultivariateNormal(mean_x, covar_x)


class BotorchCompatibleGP(
    Model, AbstractBaseModel, botorch.models.model.FantasizeMixin
):
    def __init__(self, dim_input, device, batch_size: int = 64):
        super().__init__()

        self.device = device
        self.data_dim = dim_input
        self.batch_size = batch_size

        self.likelihood = GaussianLikelihood().to(device)

        # Initialize the GP models
        self.noise_gp = SparseGPModel(None, None, self.likelihood, device)
        self.neural_gp = NeuralGPModel(None, None, self.likelihood, dim_input, device)
        self.sum_gp = SumGPModel(
            self.neural_gp, self.noise_gp, None, None, device
        )  # No separate training data

        # Initialization
        self.train_x = None
        self.train_y = None
        self.return_samples = False

    def forward(self, x):
        return self.sum_gp(x.to(self.device))

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
        train_x_tensor = torch.tensor(
            np.array(train_x.get_data()), dtype=torch.float32, device=self.device
        ).squeeze(0)
        train_y_tensor = torch.tensor(
            np.array(train_y.get_data()), dtype=torch.float32, device=self.device
        ).squeeze(0)
        train_y_tensor = train_y_tensor.squeeze(1)

        # Set training data for each model
        self.neural_gp.set_train_data(train_x_tensor, train_y_tensor, strict=False)
        self.noise_gp.set_train_data(train_x_tensor, train_y_tensor, strict=False)
        self.sum_gp.set_train_data(train_x_tensor, train_y_tensor, strict=False)

        # Ensure models are on the same device
        self.neural_gp = self.neural_gp.to(self.device)
        self.noise_gp = self.noise_gp.to(self.device)
        self.sum_gp = self.sum_gp.to(self.device)

        self.num_samples = train_y_tensor.size(0)

        # Set the training data for SumGPModel
        self.sum_gp.set_train_data(train_x_tensor, train_y_tensor, strict=False)

        # Training loop
        self.sum_gp.train()
        mll_sum = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.sum_gp)
        optimizer_sum = torch.optim.Adam(self.sum_gp.parameters(), lr=0.01)

        for i in range(50):
            optimizer_sum.zero_grad()
            output_sum = self.sum_gp(train_x_tensor)
            loss_sum = -mll_sum(output_sum, train_y_tensor).sum()
            loss_sum.backward()
            optimizer_sum.step()
            print(
                f"Epoch {i}, Loss: {loss_sum.item()}"
            )  # Use .item() to print the loss value

        return self

    def predict(
        self,
        dataset_x: AbstractDataSource,
        batch_size: int = 256,
        row_names: List[AnyStr] = None,
        return_samples=True,
    ) -> List[np.ndarray]:
        if isinstance(dataset_x, torch.Tensor):
            x_tensor = dataset_x
        else:
            # If not, assume it's a data source with a get_data() method
            try:
                x_tensor = torch.tensor(
                    np.array(dataset_x.get_data()),
                    dtype=torch.float32,
                    device=self.device,
                )
            except AttributeError:
                raise TypeError(
                    "dataset_x must be a tensor or have a get_data() method"
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
            # Concatenate all sample arrays
            concatenated_samples = np.concatenate(all_samples, axis=0)

            # Check if the array is already 2D, if not, reshape it
            if concatenated_samples.ndim == 1:
                concatenated_samples = concatenated_samples.reshape(
                    -1, 1
                )  # Reshape to 2D if necessary

            # Add an extra dimension if needed
            if concatenated_samples.ndim == 2:
                # Add an extra dimension to match the expected shape
                concatenated_samples = np.expand_dims(concatenated_samples, axis=-1)
            return concatenated_samples

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> "BotorchCompatibleGP":
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
        self.sum_gp.eval()
        mvn = self.sum_gp(X)
        if observation_noise and isinstance(self.likelihood, GaussianLikelihood):
            mvn = MultivariateNormal(
                mvn.mean, mvn.lazy_covariance_matrix.add_diag(self.likelihood.noise)
            )
        return GPyTorchPosterior(mvn)

    @staticmethod
    def get_save_file_extension() -> str:
        """
        Return the file extension for saving the model.

        Returns:
            str: The file extension.
        """
        return "pt"

    @classmethod
    def load(cls, file_path: str) -> "BotorchCompatibleGP":
        """
        Load the model from the specified file path.

        Parameters:
            file_path (str): The path from where the model should be loaded.

        Returns:
            BotorchCompatibleGP: The loaded model.
        """
        print("Loading model...")
        state_dict = torch.load(
            file_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        device = state_dict["device"]

        # Create a new model instance
        model = cls(
            state_dict["data_dim"], state_dict["device"], state_dict["batch_size"]
        )

        # Restore the state of the models and the likelihoods
        model.neural_gp.load_state_dict(state_dict["neural_gp"])
        model.noise_gp.load_state_dict(state_dict["noise_gp"])
        model.likelihood.load_state_dict(state_dict["likelihood"])
        model.sum_gp = SumGPModel(model.neural_gp, model.noise_gp, None, None, device)
        return model

    def save(self, file_path: str):
        """
        Save the model to the specified file path.

        Parameters:
            file_path (str): The path to where the model should be saved.
        """
        print("Saving model...")
        state_dict = {
            "data_dim": self.data_dim,
            "device": self.device.type,  # Save the device type (cpu or cuda)
            "batch_size": self.batch_size,
            "neural_gp": self.neural_gp.state_dict(),
            "noise_gp": self.noise_gp.state_dict(),
            "likelihood": self.likelihood.state_dict(),
        }
        torch.save(state_dict, file_path)
