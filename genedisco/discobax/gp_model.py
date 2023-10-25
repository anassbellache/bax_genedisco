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
    Type,
    List,
)

import botorch
import gpytorch
import numpy as np
import torch
import torch.optim
from botorch.models import ApproximateGPyTorchModel
from botorch.models.model import Model, TFantasizeMixin
from botorch.posteriors import Posterior, GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import VariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from slingpy import AbstractBaseModel, AbstractDataSource
from torch import Tensor
from torch.nn import Module


class VariationalGPModel(VariationalGP, botorch.models.model.FantasizeMixin):
    def __init__(self, likelihood, dim_input, device, num_inducing_points=10):
        # Initialize inducing points
        inducing_points = torch.randn(num_inducing_points, dim_input).to(device)

        # Setup the variational distribution
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0)).to(device)

        # Initialize the variational strategy
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        # Call superclass's __init__ method with the necessary arguments
        super(VariationalGPModel, self).__init__(variational_strategy=variational_strategy)

        # Define mean and covariance functions
        self.mean_module = gpytorch.means.ZeroMean().to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(device)
        self.likelihood = likelihood

        # Store device for further use
        self.device = device

    def forward(self, x):
        # Return the predictive mean and variance for the given inputs x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, **kwargs):
        return self.forward(x)

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


class NeuralGPModel(VariationalGP, botorch.models.model.FantasizeMixin):
    def __init__(self, likelihood, dim_input, device, feature_selection=100, num_inducing_points=10):
        # Initialize inducing points
        inducing_points = torch.randn(num_inducing_points, dim_input).to(device)

        # Setup the variational distribution
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0)).to(device)

        # Initialize the variational strategy
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        # Call superclass's __init__ method with the necessary arguments
        super(NeuralGPModel, self).__init__(variational_strategy=variational_strategy)
        # Define mean and covariance functions
        self.mean_module = gpytorch.means.ZeroMean().to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(device)
        self.likelihood = likelihood

        # Store device for further use
        self.device = device

        # Neural network feature extractor
        self.feature_extractor = LargeFeatureExtractor(dim_input, feature_selection).to(device)

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

    def forward(self, x):
        # Return the predictive mean and variance for the given inputs x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, **kwargs):
        return self.forward(x)


class SumGPModel(VariationalGP, botorch.models.model.FantasizeMixin):
    def __init__(self, neural_gp, noise_gp, neural_likelihood, noise_likelihood):
        super().__init__(neural_gp.variational_strategy)
        self.neural_gp = ApproximateGPyTorchModel(neural_gp, neural_likelihood)
        self.noise_gp = ApproximateGPyTorchModel(noise_gp, noise_likelihood)

    def forward(self, x):
        neural_out = self.neural_gp(x)
        noise_out = self.noise_gp(x)
        mean_x = neural_out.mean + noise_out.mean
        covar_x = neural_out.lazy_covariance_matrix + noise_out.lazy_covariance_matrix
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, **kwargs):
        return self.forward(x)

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        """Get the posterior from the sum_gp."""
        neural_posterior = self.neural_gp.posterior(X, observation_noise=observation_noise, **kwargs)
        noise_posterior = self.noise_gp.posterior(X, observation_noise=observation_noise, **kwargs)
        mean_x = neural_posterior.mean + noise_posterior.mean
        covar_x = neural_posterior.lazy_covariance_matrix + noise_posterior.lazy_covariance_matrix
        return GPyTorchPosterior(MultivariateNormal(mean_x, covar_x))

    def condition_on_observations(self: TFantasizeMixin, X: Tensor, Y: Tensor, **kwargs: Any) -> TFantasizeMixin:
        # Condition both the neural and noise GPs on the observations
        conditioned_neural_gp = self.neural_gp.condition_on_observations(X, Y, **kwargs)
        conditioned_noise_gp = self.noise_gp.condition_on_observations(X, Y, **kwargs)

        # Create a new SumGPModel with the conditioned models
        conditioned_sum_gp = SumGPModel(
            neural_gp=conditioned_neural_gp.model,
            noise_gp=conditioned_noise_gp.model,
            neural_likelihood=self.neural_gp.likelihood,
            noise_likelihood=self.noise_gp.likelihood
        )
        return conditioned_sum_gp

    def transform_inputs(self, X: Tensor, input_transform: Optional[Module] = None) -> Tensor:
        # This is a placeholder. You'll need to implement this depending on what you want to achieve.
        return X


class BotorchCompatibleGP(Model, AbstractBaseModel, botorch.models.model.FantasizeMixin):
    def __init__(self, dim_input, device, batch_size: int = 64):
        super().__init__()

        self.noise_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.noise_gp = VariationalGPModel(self.noise_likelihood, dim_input, device).float()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.neural_gp = NeuralGPModel(self.likelihood, dim_input, device).float()

        # Combined GP model
        self.sum_gp = SumGPModel(self.neural_gp, self.noise_gp, self.likelihood, self.noise_likelihood)

        self.device = device
        self.num_samples = 100
        self.return_samples = False
        self.data_dim = dim_input
        self.batch_size = batch_size
        self.sum_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # Initialize train_x and train_y as None
        self.train_x = None
        self.train_y = None

    def predict(self, dataset_x: AbstractDataSource, batch_size: int = 256, row_names: List[AnyStr] = None) -> List[
        np.ndarray]:
        x_tensor = torch.tensor(np.array(dataset_x.get_data()), dtype=torch.float32, device=self.device)
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
        train_x = torch.tensor(np.array(train_x.get_data()), dtype=torch.float32, device=self.device).squeeze(0)
        train_y = torch.tensor(np.array(train_y.get_data()), dtype=torch.float32, device=self.device).squeeze(0)
        self.num_samples = train_y.size(0)

        self.sum_gp.train()
        mll_sum = gpytorch.mlls.VariationalELBO(self.sum_likelihood, self.sum_gp, train_y.numel())
        optimizer_sum = torch.optim.Adam(self.sum_gp.parameters(), lr=0.01)
        loss_sum = 0
        for i in range(50):
            for j in range(train_x.shape[0]):
                optimizer_sum.zero_grad()
                output_sum = self.sum_gp(train_x[j])
                loss_sum = -mll_sum(output_sum, train_y[j]).sum()
                loss_sum.backward(retain_graph=True)
                optimizer_sum.step()
            print("Epoch {} , Loss: {}".format(i, loss_sum))

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
        return 'pt'

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
