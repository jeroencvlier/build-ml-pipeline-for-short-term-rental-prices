#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import os
import argparse
import logging
import wandb
import pandas as pd
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Downloaded artifact to: %s", artifact_local_path)
    df = pd.read_csv(artifact_local_path, index_col=0)

    logger.info("Removing outliers...")
    df = df[df['price'].between(args.min_price, args.max_price)]

    logger.info("Converting last_review to datetime format...")
    idx = df['longitude'].between(-74.25, -
                                  73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # save in temp directory
    logger.info(
        "Saving cleaned data to temporary directory and uploading to W&B.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        df.to_csv(os.path.join(tmp_dir, args.output_artifact))
        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(os.path.join(tmp_dir, args.output_artifact))
        run.log_artifact(artifact)

        # wait for artifact to be uploaded
        logger.info("Waiting for artifact to be uploaded...")
        artifact.wait()

    logger.info("Complete. Shutting down mlflow run!")
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Fully qualified name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=int,
        help="Minimum $ price to exclude as an outlier",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="Maximum $ price to exclude as an outlier",
        required=True
    )

    args = parser.parse_args()

    go(args)
