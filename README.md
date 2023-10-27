# GeneDisco: A benchmark for active learning in drug discovery

![Python version](https://img.shields.io/badge/Python-3.8-blue)
![Library version](https://img.shields.io/badge/Version-1.0.5-blue)

In vitro cellular experimentation with genetic interventions, using for example CRISPR technologies, is an essential 
step in early-stage drug discovery and target validation that serves to assess initial hypotheses about causal 
associations between biological mechanisms and disease pathologies. With billions of potential hypotheses to test, 
the experimental design space for in vitro genetic experiments is extremely vast, and the available experimental 
capacity - even at the largest research institutions in the world - pales in relation to the size of this biological 
hypothesis space. 

[GeneDisco (published at ICLR-22)](https://arxiv.org/abs/2110.11875) is a benchmark suite for evaluating active learning algorithms for experimental design in drug discovery. 
GeneDisco contains a curated set of multiple publicly available experimental data sets as well as open-source i
mplementations of state-of-the-art active learning policies for experimental design and exploration.

## GeneDisco ICLR-22 Challenge

Learn more about the GeneDisco challenge for experimental design for optimally exploring the vast genetic intervention space [here](https://www.gsk.ai/genedisco-challenge/).

## Install

```bash
pip install genedisco
```

## Use

### How to Run the Full Benchmark Suite?

Experiments (all baselines, acquisition functions, input and target datasets, multiple seeds) included in GeneDisco can be executed sequentially for e.g. acquired batch size `64`, `8` cycles and a `bayesian_mlp` model using:
```bash
run_experiments \
  --cache_directory=/path/to/genedisco_cache  \
  --output_directory=/path/to/genedisco_output  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --max_num_jobs=1
```
Results are written to the folder at `/path/to/genedisco_cache`, and processed datasets will be cached at `/path/to/genedisco_cache` (please replace both with your desired paths) for faster startup in future invocations.


Note that due to the number of experiments being run by the above command, we recommend execution on a compute cluster.<br/>
The GeneDisco codebase also supports execution on slurm compute clusters (the `slurm` command must be available on the executing node) using the following command and using dependencies in a Python virtualenv available at `/path/to/your/virtualenv` (please replace with your own virtualenv path):
```bash
run_experiments \
  --cache_directory=/path/to/genedisco_cache  \
  --output_directory=/path/to/genedisco_output  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --schedule_on_slurm \
  --schedule_children_on_slurm \
  --remote_execution_virtualenv_path=/path/to/your/virtualenv
```

Other scheduling systems are currently not supported by default.

### How to Run A Single Isolated Experiment (One Learning Cycle)?

To run one active learning loop cycle, for example, with the `"topuncertain"` acquisition function, the `"achilles"` feature set and
the `"schmidt_2021_ifng"` task, execute the following command:
```bash
active_learning_loop  \
    --cache_directory=/path/to/genedisco/genedisco_cache \
    --output_directory=/path/to/genedisco/genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="topuncertain" \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
```


# New Additions: Incorporating InfoBax algorithms: For that, we add three components: algorithm, acquisition, and gp_model 

### How to Evaluate a Custom Acquisition Function?

To run disco bax acquisition function.
```bash
active_learning_loop  \
    --cache_directory=/path/to/bax_genedisco/cache \
    --output_directory=/path/to/bax_genedisco/output \
    --model_name="base_gp" \
    --acquisition_function_name="disco_bax" \
    --acquisition_batch_size=10 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng"
```


### Algorithm

#### Overview:

`Algorithm` serves as a base class for a variety of optimization algorithms. Its main purpose is to provide a common framework and reusable methods for different algorithmic approaches. It leverages the concept of an "execution path", a series of points visited during the algorithm's execution.

#### Key Methods:

- `initialize()`: Resets the algorithm's state, especially the execution path.
- `get_next_x()`: Determines the next point, x, the algorithm should evaluate. By default, it chooses random points until the execution path has 10 elements.
- `take_step(f)`: Executes a single step of the algorithm by determining a new point and evaluating it.
- `run_algorithm_on_f(f)`: Runs the entire algorithm on a given function.
- `get_output()`: An abstract method to retrieve the output of the algorithm, which child classes must implement.
- `get_copy()`: Returns a deep copy of the algorithm instance.
- `get_exe_path_crop()`: Retrieves the current execution path.
- `get_output_dist_fn()`: Provides a default distance function for pairs of outputs.

---

# Documentation for the SubsetSelect class

### SubsetSelect (Derived from Algorithm)

#### Overview:

`SubsetSelect` is an implementation of a greedy subset selection algorithm. The primary goal is to select a subset of `k` points from a given dataset `X` that optimizes a specific criteria, in this case, based on Monte Carlo estimated scores. It's especially useful in scenarios where one wants to choose a representative subset from a larger set without evaluating the entire set.

#### Key Attributes:

- `k`: Number of subset points to select.
- `X`: A finite sample set from which to select the subset.
- `selected_subset`: List storing the currently selected points.
- `mc_samples`: Number of Monte Carlo samples used for expectation estimation.
- `input_dim`: Dimensionality of the data.
- `device`: The computing device (CPU/GPU) on which calculations should run.
- `exe_path`: The current execution path (history of selected points).

#### Key Methods:

- `initialize()`: Resets the state, primarily the execution path and selected subset.
- `monte_carlo_expectation(...)`: Estimates scores for given candidates using Monte Carlo sampling. It parallelizes computation for efficiency.
- `handle_first_selection(f)`: Manages the first point selection based on the posterior mean from a GP.
- `select_next(f)`: Chooses the next best point to add to the subset, based on Monte Carlo scores.
- `take_step(f)`: Executes a single step of the subset selection process.
- `update_exe_paths(x, y)`: Updates the execution path with a new point and its associated value.
- `get_output()`: Retrieves the current list of selected points.
- `get_exe_paths(f)`: Produces the execution paths for different possible sequences of subset selection.

---

### Global Architecture and Connection to Greedy Subset Selection:

The primary class, `Algorithm`, encapsulates general optimization methods, but does not prescribe a specific approach. This abstraction allows derived classes, like `SubsetSelect`, to leverage these general methods while implementing specific logic unique to a particular algorithm.

`SubsetSelect` uses a greedy approach. In each step, it selects the next point that maximizes a given criterion. This criterion is based on Monte Carlo estimations of the potential value of adding each candidate point to the current subset. By always choosing the point with the highest estimated value in each step, the algorithm ensures it is always making the locally optimal choice. This greedy nature guarantees a degree of optimality for the resulting subset, but not necessarily global optimality. However, the Monte Carlo approximation ensures the approach is efficient and scalable, especially when dealing with large datasets.

This modular architecture, where specific algorithmic logic is isolated in derived classes, ensures clean, maintainable code. It also makes it easier to implement and test new algorithms in the future.

Sure, let's provide a more consolidated documentation by incorporating the explanations of the new classes (`VariationalGPModel`, `LargeFeatureExtractor`, `NeuralGPModel`, and `SumGPModel`) with the previously provided documentation:

---

# GP Models Documentation

## Table of Contents:
1. Background
2. Models
    - ApproximateGPyTorchModel
    - VariationalGPModel
    - LargeFeatureExtractor
    - NeuralGPModel
    - SumGPModel
3. Usage
4. Conclusion

## 1. Background
The provided classes define different Gaussian Process (GP) model structures. GPs are a set of flexible non-parametric methods used for regression and classification tasks. They can be combined with deep learning architectures and other strategies to provide scalable and efficient modeling of complex data.

## 2. Models

### ApproximateGPyTorchModel
This model integrates `BoTorch` and `GPyTorch` libraries to build Gaussian processes. It offers functionalities such as getting the posterior and conditioning on new observations.

### VariationalGPModel
- **Purpose**: Implementation of a variational GP model for scalable Bayesian optimization.
- **Key Components**:
    - **Inducing Points**: Represent the GP and make it scalable.
    - **Variational Strategy**: How the GP is approximated using the inducing points.
    - **FantasizeMixin**: Allows the model to be conditioned on hypothetical/fantasy observations.

### LargeFeatureExtractor
- **Purpose**: Neural network-based transformation of raw data into informative features.
- **Key Components**:
    - **Layers**: Deep neural network for feature extraction.

### NeuralGPModel
- **Purpose**: Combines deep learning and Gaussian Processes.
- **Key Components**:
    - **FeatureExtractor**: Uses `LargeFeatureExtractor` to transform raw data.
    - **Update Train Data**: Allows for dynamic updating of training data based on extracted features.

### SumGPModel
- **Purpose**: Represents a GP model that sums outputs from two other GP models: a neural GP and a noise GP.
- **Key Components**:
    - **Neural GP**: Captures underlying structure or signal.
    - **Noise GP**: Captures noise or unexplained variance.

## 3. Usage
To use these models:
1. Initialize the desired model with the required parameters.
2. Pass the training data to the model.
3. Make predictions on test data.
4. If needed, condition the model on new observations to refine future predictions.


## 4. Conclusion
These classes provide a powerful toolkit for building and using Gaussian Process models in various configurations, including hybrid models that combine the strengths of deep learning and Gaussian Processes.

---


# Documentation for the BotorchCompatibleGP class

## Overview

The `BotorchCompatibleGP` class provides a Gaussian Process (GP) model compatible with the BoTorch  library. The Gaussian Process is a popular machine learning tool, especially for regression tasks, because of its non-parametric nature, meaning it can flexibly adapt to the structure in the data without a predefined fixed form. 

In many scientific and engineering domains, models need to function efficiently in high-dimensional spaces, possibly involving big data. Traditional GP models, while effective, can be computationally challenging in these high-dimensional big data scenarios. This class addresses such challenges by offering a more scalable GP model using BoTorch and GPyTorch, libraries specialized in Bayesian optimization and Gaussian Processes, respectively.

## Key Components

### 1. `VariationalGPModel` and `NeuralGPModel`:

These are assumed to be separate implementations of GP models that might differ in their architecture or functioning. The exact implementation details might vary, but in general:

- `VariationalGPModel`: Usually represents a scalable approach to GPs, allowing them to handle larger datasets by approximating the true posterior with a variational distribution.
  
- `NeuralGPModel`: Often combines neural networks with GPs, enabling the model to capture complex patterns in data and generalize across various tasks.

### 2. `SumGPModel`:

This is a combination of the aforementioned `VariuralGPModel` and `NeuralGPModel`, enabling the GP to have both scalability and flexibility in modeling.

### 3. `SubsetSelect Algorithm`:

The reference to the subset selection algorithm in the class, although not explicitly present, can be linked with how the model is trained and used. Subsetting or sparse approximations are strategies that can make GPs more computationally tractable. The subsets of data points, often called inducing points, help in approximating the full GP, making it feasible to use in large datasets.

### 4. FantasizeMixin:

This mixin from BoTorch provides the capability to create "fantasy" models. In the context of Bayesian optimization, fantasy models are useful to anticipate the effect of potential future observations when selecting the next points to evaluate.

## Methods Overview:

1. **predict**: Predicts the outputs for given inputs. Handles large datasets by splitting them into batches and processing them separately. Can also return samples from the predictive distribution.

2. **fit**: Trains the GP model on the provided dataset.

3. **load**: Loads a previously saved model from a file.

4. **save**: Saves the current model to a file.

5. **condition_on_observations**: Creates a new GP model conditioned on additional observations.

6. **posterior**: Returns the posterior distribution of the GP for the given inputs.

7. **forward**: A method to get predictions from the model. This is typically used when the model is being used within a larger framework or when combined with other models.

## Using BotorchCompatibleGP:

This model can be used as a replacement for standard GPs in any application where scalability and efficiency in high-dimensional settings with big data are crucial. Especially when using BoTorch for Bayesian optimization tasks, this GP can be integrated seamlessly, offering better computational performance and modeling flexibility. 

When integrating with BoTorch, this GP model can leverage various acquisition functions provided by BoTorch, and due to its compatibility with the FantasizeMixin, it can also be used in algorithms that require constructing fantasy models. 

# Documentation for the `DiscoBAXAdditive` Acquisition Function

## Overview

`DiscoBAXAdditive` is an acquisition function designed specifically to work with the `slingpy` and `genedisco` libraries. It is implemented to help with the process of selecting the most informative samples in a Bayesian Active Learning framework. The core idea behind this acquisition function is the calculation of the Expected Information Gain (EIG) of potential samples, which represents the expected reduction in uncertainty when the samples are added to the training set.

## Process Flow

The function operates in several key steps:

1. **Sampling Many Paths**: Execution paths are sampled to capture different potential sequences of acquiring points. Each execution path represents a possible sequence in which the model could select data points.

2. **Subset Select**: It makes use of a subset selection algorithm (`SubsetSelect`), which provides a set of execution paths based on the given dataset and model. These paths form the foundation for which the information gain is calculated.

3. **Expected Information Gain (EIG)**: For each execution path, a fantasy model is created. A fantasy model is essentially a simulation: it pretends that a specific set of points has been observed, and then makes predictions based on that. The EIG is computed using both the current model and the fantasy models. 

## Example Usage

```python
>>> model = SingleTaskGP(train_X, train_Y)
>>> EIG = ExpectedInformationGain(model, algo, algo_params)
>>> eig = EIG(test_X)
```

## Detailed Breakdown of Parameters and Methods:

### Constructor:

#### Args:
- `path_sample_num`: The number of execution paths to sample. Defaults to `10`.

### `__call__` Method:

This method is the core of the acquisition function and is used to nominate experiments for the next learning round using Expected Information Gain.

#### Args:

- `dataset_x`: The dataset containing all training samples.
  
- `batch_size`: Size of the batch to acquire. Determines how many samples you'd want to select in this round of active learning.
  
- `available_indices`: A list containing the indices (or names) of the samples that were not chosen in previous rounds.
  
- `last_selected_indices`: The list of indices that were selected in the previous cycle. Useful for ensuring no repetition of samples.
  
- `last_model`: The predictive model trained using labeled samples chosen in the prior rounds.

#### Returns:

- A list containing the indices (or names) of the samples chosen for the next learning round.

## Notes:

- This class relies heavily on the `slingpy` and `genedisco` libraries, as well as the `BoTorch` library's `fantasize` method for constructing fantasy models.
  
- When considering the application of this acquisition function in different scenarios or with different datasets, ensure that the underlying assumptions and characteristics of the function align with the requirements and nature of the specific problem.

- Always ensure that the `slingpy`, `genedisco`, and other required libraries are properly installed and imported before using this class.

## Citation

Please consider citing, if you reference or use our methodology, code or results in your work:

    @inproceedings{mehrjou2022genedisco,
        title={{GeneDisco: A Benchmark for Experimental Design in Drug Discovery}},
        author={Mehrjou, Arash and Soleymani, Ashkan and Jesson, Andrew and Notin, Pascal and Gal, Yarin and Bauer, Stefan and Schwab, Patrick},
        booktitle={{International Conference on Learning Representations (ICLR)}},
        year={2022}
    }

### License

[License](LICENSE.txt)

### Authors

Patrick Schwab, GlaxoSmithKline plc<br/>
Arash Mehrjou, GlaxoSmithKline plc<br/>
Andrew Jesson, University of Oxford<br/>
Ashkan Soleymani, MIT

### Acknowledgements

PS and AM are employees and shareholders of GlaxoSmithKline plc.

https://github.com/BlackHC/batchbald_redux contributed code for ConsistentMCDropout.
