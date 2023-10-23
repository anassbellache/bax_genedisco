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

from typing import (
    Any,
    AnyStr,
    Type,
    List,
    Optional,
)

import botorch
import gpytorch
import numpy as np
import torch
import torch.optim
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskVariationalGP
from botorch.models.model import Model, TFantasizeMixin
from botorch.posteriors import Posterior, GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution
from slingpy import AbstractBaseModel, AbstractDataSource
from torch import Tensor
from torch.nn import Module


class VariationalGPModel(botorch.models.SingleTaskVariationalGP, botorch.models.model.FantasizeMixin):
    def __init__(self, train_X, train_Y, likelihood, dim_input, device, num_inducing_points=10):
        # Initialize inducing points
        inducing_points = torch.randn(num_inducing_points, dim_input).to(device)

        # Setup the variational distribution
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0)).to(device)

        # Define mean and covariance functions
        mean_module = gpytorch.means.ZeroMean().to(device)
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(device)

        # Call superclass's __init__ method with the necessary arguments
        super(VariationalGPModel, self).__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            covar_module=covar_module,
            mean_module=mean_module,
            learn_inducing_points=True
        )

        # Store device for further use
        self.device = device
        # The rest of the class remains unchanged...

    def posterior(self, X: Tensor, *args, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        # Get the predictive distribution (prior) for X
        predictive_distribution = self.forward(X)

        # Get the approximate posterior using the variational strategy
        posterior_distribution = self.variational_strategy(X, predictive_distribution)

        # If observation_noise flag is set, add the likelihood noise to the posterior
        if observation_noise:
            posterior_distribution = self.likelihood(posterior_distribution, *args, **kwargs)

        # Return the posterior as a Botorch GPyTorchPosterior object
        return GPyTorchPosterior(posterior_distribution)

    def condition_on_observations(self: TFantasizeMixin, X: Tensor, Y: Tensor, **kwargs: Any) -> TFantasizeMixin:
        # Combine the original inducing points with the new observations
        fantasy_inducing_points = torch.cat([self.variational_strategy.inducing_points, X], dim=0)

        # Create a new model with the updated set of inducing points
        fantasy_model = self.__class__(
            input_dim=X.size(-1),
            likelihood=self.likelihood,
            device=self.device,
            num_inducing_points=fantasy_inducing_points.size(0)
        )

        # Set the inducing points of the new model to the combined set of points
        fantasy_model.variational_strategy.inducing_points = fantasy_inducing_points

        # Return the new fantasy model
        return fantasy_model

    def transform_inputs(self, X: Tensor, input_transform: Optional[Module] = None) -> Tensor:
        pass

    def __call__(self, x, **kwargs):
        return self.forward(x)


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
            torch.nn.Linear(50, feature_dim)
        )

    def forward(self, x):
        # Pass through the layers
        x = self.layers(x)
        return x


class NeuralGPModel(SingleTaskVariationalGP, botorch.models.model.FantasizeMixin):
    def __init__(self, train_X, train_Y, likelihood, device, feature_dim=100, num_inducing_points=100):
        # Inducing points setup
        inducing_points = torch.randn(num_inducing_points, train_X.size(-1)).to(device)

        # Variational distribution
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points).to(device)

        # Covariance and mean functions
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(device)
        mean_module = gpytorch.means.ConstantMean().to(device)

        # Superclass initialization
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            covar_module=covar_module,
            mean_module=mean_module,
            learn_inducing_points=True
        )

        # Neural network feature extractor
        self.feature_extractor = LargeFeatureExtractor(train_X.size(-1), feature_dim).to(device)

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
            train_inputs_projected = self.feature_extractor(self.train_inputs[0]).to(self.device)

            # Concatenate old and new data
            if train_inputs_projected.dim() == 1:
                updated_train_x = torch.cat([train_inputs_projected, new_x_projected.squeeze(0)], dim=0).to(self.device)
            else:
                updated_train_x = torch.cat([train_inputs_projected, new_x_projected], dim=0).to(self.device)

            updated_train_y = torch.cat([self.train_targets, new_y], dim=0).to(self.device)

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

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        # Process the input through the neural network.
        # Obtain the prior distribution.
        mvn = self(X)

        # If observation noise should be added and the likelihood is GaussianLikelihood
        if observation_noise and isinstance(self.likelihood, GaussianLikelihood):
            noise = self.likelihood.noise
            mvn = MultivariateNormal(mvn.mean, mvn.lazy_covariance_matrix.add_diag(noise))

        # Return the botorch wrapper around GPyTorch's posterior.
        return GPyTorchPosterior(mvn)

    def transform_inputs(self, X: Tensor, input_transform: Optional[Module] = None) -> Tensor:
        pass

    def __call__(self, x):
        return self.forward(x)


class SumGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, neural_gp, noise_gp):
        super().__init__(neural_gp.variational_strategy)
        self.neural_gp = neural_gp
        self.noise_gp = noise_gp

    def forward(self, x):
        neural_out = self.neural_gp(x)
        noise_out = self.noise_gp(x)
        mean_x = neural_out.mean + noise_out.mean
        covar_x = neural_out.lazy_covariance_matrix + noise_out.lazy_covariance_matrix

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BotorchCompatibleGP(Model, AbstractBaseModel, botorch.models.model.FantasizeMixin):
    def __init__(self, dim_input, device, batch_size: int = 64):
        super().__init__()
        train_X = torch.zeros(dim_input, device=device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.neural_gp = NeuralGPModel(train_X, None, self.likelihood, device).float()

        self.noise_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.noise_gp = VariationalGPModel(train_X, None, self.noise_likelihood, dim_input, device).float()

        # Combined GP model
        self.sum_gp = SumGPModel(self.neural_gp, self.noise_gp)

        self.device = device
        self.num_samples = 100
        self.return_samples = False
        self.data_dim = dim_input
        self.batch_size = batch_size

        # Initialize train_x and train_y as None
        self.train_x = None
        self.train_y = None

    def predict(self, dataset_x: AbstractDataSource, batch_size: int = 256, row_names: List[AnyStr] = None) -> List[
        np.ndarray]:
        x_tensor = torch.tensor(dataset_x.get_data(), dtype=torch.float32, device=self.device)
        self.sum_gp.eval()
        self.likelihood.eval()
        self.noise_gp.eval()
        self.noise_likelihood.eval()

        # Split the data into batches
        num_samples = x_tensor.size(0)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        all_pred_means = []
        all_pred_stds = []
        all_samples = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(num_batches):
                start_i = i * self.batch_size
                end_i = min((i + 1) * self.batch_size, num_samples)
                batch_x = x_tensor[start_i:end_i]

                main_pred = self.likelihood(self.sum_gp(batch_x.to(self.device)))

                combined_mean = main_pred.mean
                combined_stddev = main_pred.variance
                all_pred_means.append(combined_mean.cpu().numpy())
                all_pred_stds.append(combined_stddev.cpu().numpy())

                # Sample from the predictive distribution if required
                if self.return_samples:
                    # Since sampling from the sum of two GP posteriors isn't straightforward,
                    # for simplicity we'll sample separately and add them.
                    main_sample = main_pred.sample(sample_shape=torch.Size([self.num_samples])).to(self.device)
                    combined_sample = main_sample

                    all_samples.append(combined_sample.cpu().numpy())

                print(f"Batch {i}: {combined_mean.shape}")

        # Concatenate results from all batches
        pred_mean = np.concatenate(all_pred_means, axis=0)
        pred_std = np.concatenate(all_pred_stds, axis=0)

        # Compute the 95% confidence bounds
        upper_bound = pred_mean + 1.96 * pred_std
        lower_bound = pred_mean - 1.96 * pred_std

        # Compute the margins
        y_margins = upper_bound - lower_bound

        if self.return_samples:
            samples = np.concatenate(all_samples, axis=0)
            return [pred_mean, pred_std, y_margins, samples]
        else:
            return pred_mean

    def fit(self, train_x: AbstractDataSource, train_y: Optional[AbstractDataSource] = None,
            validation_set_x: Optional[AbstractDataSource] = None,
            validation_set_y: Optional[AbstractDataSource] = None) -> "AbstractBaseModel":

        if train_x is None or train_y is None:
            raise ValueError("Both train_x and train_y must be provided")

        # Convert AbstractDataSource to torch.Tensor
        train_x = torch.tensor(train_x.get_data(), dtype=torch.float32, device=self.device)
        train_y = torch.tensor(train_y.get_data(), dtype=torch.float32, device=self.device)
        self.num_samples = train_y.size(0)

        # Train NeuralGPModel
        self.neural_gp.train()
        self.likelihood.train()
        mll_neural = gpytorch.mlls.VariationalELBO(self.likelihood, self.neural_gp, train_y.numel(),
                                                   combine_terms=False)
        optimizer_neural = torch.optim.Adam(self.neural_gp.parameters(), lr=0.01)
        # This is a basic loop for training; in real applications, consider using multiple epochs
        for i in range(50):
            optimizer_neural.zero_grad()
            output_neural = self.neural_gp(train_x)
            loss_neural = -mll_neural(output_neural, train_y)
            loss_neural.backward()
            optimizer_neural.step()

        # Train VariationalGPModel for noise
        self.noise_gp.train()
        self.noise_likelihood.train()
        mll_variational = gpytorch.mlls.VariationalELBO(self.noise_likelihood, self.noise_gp, train_y.numel())
        optimizer_noise = torch.optim.Adam(self.noise_gp.parameters(), lr=0.01)
        # Again, this is a basic loop for training
        for i in range(50):
            optimizer_noise.zero_grad()
            output_noise = self.noise_gp(train_x)
            loss_noise = -mll_variational(output_noise, train_y)
            loss_noise.backward()
            optimizer_noise.step()

        return self

    @classmethod
    def load(cls: Type["BotorchCompatibleGP"], file_path: AnyStr) -> "BotorchCompatibleGP":
        """
        Load the model from the specified file path.

        Parameters:
            - file_path (str): The path from where the model should be loaded.

        Returns:
            - model (BotorchCompatibleGP): The loaded model.
        """
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
        return '.pt'

    def save(self, file_path: AnyStr):
        """
        Save the model to the specified file path.

        Parameters:
            - file_path (str): The path to where the model should be saved.
        """
        state_dict = {
            "data_dim": self.data_dim,
            "neural_gp": self.neural_gp.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "noise_gp": self.noise_gp.state_dict(),
            "noise_likelihood": self.noise_likelihood.state_dict()
        }
        torch.save(state_dict, file_path)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        """
        Condition the NeuralGP and VariationalGP on new observations (X, Y)
        and return a new BotorchCompatibleGP with the updated models.
        """
        # Combine the old training data with the new observations
        if self.train_x is not None and self.train_y is not None:
            X = torch.cat([self.train_x, X], dim=0)
            Y = torch.cat([self.train_y, Y], dim=0)

        # Create a new instance of the model
        fantasy_model = BotorchCompatibleGP(self.data_dim, self.device)

        # Update the training data of the new model
        fantasy_model.train_x = X
        fantasy_model.train_y = Y

        # Return the fantasy model
        return fantasy_model

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        """Get the posterior from the sum_gp."""
        mvn = self.sum_gp.forward(X)

        # If observation noise should be added
        if observation_noise and isinstance(self.likelihood, GaussianLikelihood):
            noise = self.likelihood.noise
            mvn = MultivariateNormal(mvn.mean, mvn.lazy_covariance_matrix.add_diag(noise))

        return GPyTorchPosterior(mvn)

    def forward(self, x):
        # This might need modifications based on what BaseGPModel's predict method returns
        pred = self.sum_gp(x.to(self.device))
        return pred
