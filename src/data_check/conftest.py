'''Generates fixtures for the data tests'''

import logging
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    '''Adds the options for the tests'''
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    '''Returns the data as a pandas DataFrame'''
    run = wandb.init(job_type="data_tests", resume=True)
    logging.basicConfig(
        filename='./logs/pytest_logs.log',

        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    # Download input artifact.
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    logging.info('File path for the data is: %s', request.config.option.csv)

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    '''Returns the reference data as a pandas DataFrame'''
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact.
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    logging.info('File path for the reference data is: %s', data_path)
    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    '''Returns the threshold for the KL test'''
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    '''Returns the minimum price for the data'''
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    '''Returns the maximum price for the data'''
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
