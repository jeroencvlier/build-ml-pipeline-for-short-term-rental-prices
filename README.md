
# NYC Short-Term Rental Price Prediction Pipeline

## Overview
This ML pipeline predicts short-term rental prices in NYC. It automates the process of data ingestion, cleaning, testing, training, and evaluation, ensuring robust and efficient weekly updates with new data sets.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Pipeline Components](#pipeline-components)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Links
[W&B Project](https://wandb.ai/jeroencvlier/nyc_airbnb)
[GitHub Repository](https://github.com/jeroencvlier/build-ml-pipeline-for-short-term-rental-prices)

## Getting Started
### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Git](https://git-scm.com/downloads)

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/jeroencvlier/build-ml-pipeline-for-short-term-rental-prices.git
   cd build-ml-pipeline-for-short-term-rental-prices
   ```

2. **Create and Activate the Conda Environment**
   ```bash
   conda env create -f environment.yml
   conda activate nyc_airbnb_dev
   ```

3. **Initialize Weights & Biases**
   ```bash
   wandb login
   ```

## Pipeline Components
- **Data Download**: Fetches the latest rental data.
- **Basic Cleaning**: Cleans the dataset by handling outliers and missing values.
- **Data Testing**: Ensures data quality and consistency.
- **Data Splitting**: Segregates the dataset into training, validation, and test sets.
- **Model Training**: Trains a Random Forest model.
- **Hyperparameter Optimization**: Fine-tunes the model parameters.
- **Model Testing**: Evaluates the model's performance on the test set.
- **Visualization**: Provides insights into the model's performance and data flow.

## Usage
Run the entire pipeline or specific components using MLflow:
```bash
mlflow run . -P steps=step_name
```
To run with custom configurations:
```bash
mlflow run . -P hydra_options="your_configuration"
```

To retrain the model with new data:
```bash
mlflow run https://github.com/jeroencvlier/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.1 -P hydra_options="etl.sample='sample2.csv'"
```

## License
Distributed under the MIT License. See [LICENSE](LICENSE.txt) for more information.

---

Project developed as part of the [Udacity Machine Learning Operations Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).
