'''Main pipeline script for the mlflow project'''

import json
import tempfile
import os
import mlflow
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]
# NOTE: We do not include "test_regression_model" in the steps so it is not run by mistake.
# You first need to promote a model export to "prod" before you can run this,


@hydra.main(config_name='config')
def go(config: DictConfig):
    '''
    Main pipeline script for the mlflow project specified in the
    config file and initiating the steps specified in the config file
    '''
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # perform basic cleaning
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']},
            )

        if "data_check" in active_steps:
            # perform data check
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_split" in active_steps:
            # perform data split
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]})

        if "train_random_forest" in active_steps:
            # NOTE: we need to serialize the random forest configuration into
            # JSON to use for W&B
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+", encoding="utf-8") as fp:
                json.dump(
                    dict(config["modeling"]["random_forest"].items()), fp)

            # train random forest
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"},
            )

        if "test_regression_model" in active_steps:
            # test regression model, this step needs to be called mannually as
            # it requires a model to be promoted to "prod"
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                parameters={
                    "test_dataset": "test_data.csv:latest",
                    "mlflow_model": "random_forest_export:prod"})


if __name__ == "__main__":
    go()
