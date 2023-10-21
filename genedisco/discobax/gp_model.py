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
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors import Posterior, GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from slingpy import AbstractBaseModel, AbstractDataSource
from torch import Tensor, optim
from torch.nn import Module


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


class NeuralGPModel(DeepGP, botorch.models.model.FantasizeMixin):
    def __init__(self, data_dim, likelihood, device, feature_dim=100):
        super().__init__(None, None, likelihood)

        self.data_dim = data_dim
        self.device = device
        self.feature_extractor = LargeFeatureExtractor(data_dim, feature_dim).to(device)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

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

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> 'NeuralGPModel':
        """
        Condition the NeuralGP on new observations (X, Y) and return a new NeuralGPModel.
        """
        # Ensure that the new data is processed using the feature extractor
        X_projected = self.feature_extractor(X)

        # Make sure self.train_inputs[0] is the projected version
        train_inputs_projected = self.feature_extractor(self.train_inputs[0])

        if train_inputs_projected.dim() == 1:
            updated_train_x = torch.cat([train_inputs_projected, X_projected.squeeze(0)], dim=0).to(self.device)
        else:
            updated_train_x = torch.cat([train_inputs_projected, X_projected], dim=0).to(self.device)

        updated_train_y = torch.cat([self.train_targets, Y], dim=0).to(self.device)
        data_dim = updated_train_x.shape(-1)

        new_model = self.__class__(data_dim, self.likelihood, self.device)
        new_model.likelihood = self.likelihood
        new_model.mean_module = self.mean_module
        new_model.covar_module = self.covar_module
        new_model.feature_extractor = self.feature_extractor
        new_model.set_train_data(updated_train_x, updated_train_y)

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
        projected_x = self.feature_extractor(x.to(self.device))
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x):
        return self.forward(x)


class BotorchCompatibleGP(Model, AbstractBaseModel):
    def __init__(self, dim_input, device, batch_size: int = 10):
        super().__init__()
        self.num_samples = 100
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.model = NeuralGPModel(dim_input, self.likelihood, device).float()
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.return_samples = False
        self.data_dim = dim_input
        self.batch_size = batch_size

        self.noise_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.noise_gp = NeuralGPModel(None, None, self.noise_likelihood, device)
        # Initialize train_x and train_y as None
        self.train_x = None
        self.train_y = None

    def predict(self, dataset_x: AbstractDataSource, batch_size: int = 256, row_names: List[AnyStr] = None) -> List[
        np.ndarray]:
        x_tensor = torch.tensor(dataset_x.get_data(), dtype=torch.float32, device=self.device)
        self.model.eval()
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

                main_pred = self.likelihood(self.model(batch_x.to(self.device)))
                noise_pred = self.noise_likelihood(self.noise_gp(batch_x.to(self.device)))
                print("main_pred: ", main_pred.variance.device)

                combined_mean = main_pred.mean + noise_pred.mean
                combined_stddev = torch.sqrt(main_pred.variance + noise_pred.variance)

                all_pred_means.append(combined_mean.cpu().numpy())
                all_pred_stds.append(combined_stddev.cpu().numpy())

                # Sample from the predictive distribution if required
                if self.return_samples:
                    # Since sampling from the sum of two GP posteriors isn't straightforward,
                    # for simplicity we'll sample separately and add them.
                    main_sample = main_pred.sample(sample_shape=torch.Size([self.num_samples])).to(self.device)
                    noise_sample = noise_pred.sample(sample_shape=torch.Size([self.num_samples])).to(self.device)
                    combined_sample = main_sample + noise_sample

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

        if train_y is None:
            raise ValueError("train_y cannot be None")

        if train_x is None:
            raise ValueError("train_x cannot be None")

        # Convert AbstractDataSource to torch.Tensor
        train_x = torch.tensor(train_x.get_data(), dtype=torch.float32, device=self.device)
        train_y = torch.tensor(train_y.get_data(), dtype=torch.float32, device=self.device)
        self.num_samples = train_y.size(0)

        if validation_set_x and validation_set_y:
            validation_set_x = torch.tensor(validation_set_x.get_data(), dtype=torch.float32, device=self.device)
            validation_set_y = torch.tensor(validation_set_y.get_data(), dtype=torch.float32, device=self.device)

        # Remaining initializations
        noise = 1e-4
        self.likelihood.noise = noise
        self.model.train()
        self.likelihood.train()
        self.noise_gp.train()
        self.noise_likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters()},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.likelihood.parameters()},
            {'params': self.noise_gp.parameters()},
            {'params': self.noise_likelihood.parameters()}
        ], lr=0.01)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               verbose=True)

        loss_function = torch.nn.MSELoss()

        num_epochs = 50
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Predictions
            main_output = self.model(train_x)
            noise_output = self.noise_gp(train_x)

            # Combined prediction
            combined_output = main_output.mean + noise_output.mean

            # Calculate the combined loss
            main_loss = -self.likelihood(main_output, train_y).log_prob(train_y)
            noise_loss = -self.noise_likelihood(noise_output, train_y).log_prob(train_y)
            combined_loss = loss_function(combined_output, train_y)

            # Weighted sum of losses
            total_loss = main_loss + noise_loss + combined_loss
            total_loss = total_loss.mean()

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss.item()}")

            total_loss.backward()
            optimizer.step()

            # If validation set is provided, compute validation loss and pass it to scheduler
            if validation_set_x is not None and validation_set_y is not None:
                self.model.eval()
                self.noise_gp.eval()

                with torch.no_grad():
                    main_val_output = self.model(validation_set_x)
                    noise_val_output = self.noise_gp(validation_set_x)
                    combined_val_output = main_val_output.mean + noise_val_output.mean
                    val_loss = loss_function(combined_val_output, validation_set_y)

                scheduler.step(val_loss)

                self.model.train()
                self.noise_gp.train()

        self.model.eval()
        self.likelihood.eval()
        self.noise_gp.eval()
        self.noise_likelihood.eval()

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

        # Restore the state of the model and the likelihood
        model.model.load_state_dict(state_dict["model"])
        model.likelihood.load_state_dict(state_dict["likelihood"])

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
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
        }
        torch.save(state_dict, file_path)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> 'BotorchCompatibleGP':
        """
        Condition the GPyTorchCompatibleGP on new observations (X, Y) and return a new GPyTorchCompatibleGP.
        """
        if self.train_x is None or self.train_y is None:
            updated_train_x = X.to(self.device)
            updated_train_y = Y.to(self.device)
        else:
            # Combine new observations with existing training data
            updated_train_x = torch.cat([self.train_x, X], dim=0).to(self.device)
            updated_train_y = torch.cat([self.train_y, Y], dim=0).to(self.device)

        # Create a new model with the combined data
        data_dim = updated_train_x.shape[-1]
        new_model = BotorchCompatibleGP(data_dim, self.device)

        # Ensure that the new model carries over the necessary attributes
        # You might have to adjust the attributes being copied based on your needs
        new_model.model = self.model
        new_model.likelihood = self.likelihood
        new_model.noise_gp = self.noise_gp
        new_model.noise_likelihood = self.noise_likelihood

        # Set the training data for the new model
        new_model.train_x = updated_train_x
        new_model.train_y = updated_train_y

        return new_model

    def posterior(
            self,
            X: Tensor,
            output_indices: Optional[List[int]] = None,
            observation_noise: bool = False,
            posterior_transform: Optional[PosteriorTransform] = None,
            **kwargs: Any,
    ) -> Posterior:

        # Ensure the model is in evaluation mode
        self.eval()

        # If your model or noise_gp have any input transformations, apply them here
        # X_transformed = self.transform_inputs(X)
        # For simplicity, I'm assuming no transformations:
        X_transformed = X.to(self.device)

        # Compute the primary function posterior from self.model
        function_posterior = self.model.posterior(X_transformed)

        if observation_noise:
            # If observation noise is considered, compute the noise posterior from self.noise_gp
            noise_posterior = self.noise_gp.posterior(X_transformed, **kwargs)

            # Manually combine the mean and covariance of the two posteriors
            combined_mean = function_posterior.mean + noise_posterior.mean
            combined_covar = function_posterior.mvn.covariance_matrix + noise_posterior.mvn.covariance_matrix

            # Construct a new posterior with the combined mean and covariance
            combined_mvn = torch.distributions.MultivariateNormal(combined_mean, combined_covar)
            combined_posterior = GPyTorchPosterior(combined_mvn)

            # If you have any posterior transforms, apply them here
            if posterior_transform is not None:
                combined_posterior = posterior_transform(combined_posterior)

            return combined_posterior

        # If observation noise is not considered, return only the function posterior
        # If you have any posterior transforms, apply them here
        if posterior_transform is not None:
            function_posterior = posterior_transform(function_posterior)

        return function_posterior

    def forward(self, x):
        # This might need modifications based on what BaseGPModel's predict method returns
        pred = self.predict(x.to(self.device))
        return pred
