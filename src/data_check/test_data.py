import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    """
    Test that the column names are the expected ones and in the right order
    """

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # Unordered check
    # get the difference between the two sets

    assert set(expected_colums) == set(
        these_columns), f"The difference between the columns are: {set(expected_colums).difference(set(these_columns))}"

    # This also enforces the same order
    assert list(expected_colums) == list(
        these_columns), f"Columns are not ordered as expected, {these_columns} in the data but {expected_colums} from the reference dataset!"


def test_neighborhood_names(data):
    """
    Test that the neighborhood names are the expected ones (unordered)
    """

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -
                                    73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################
def test_row_count(data):
    """
    Test that the number of rows in the cleaned dataset is within the expected range
    """
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    """
    Test that the price column does not contain values outside the specified range
    """
    assert data['price'].between(min_price, max_price).all()
