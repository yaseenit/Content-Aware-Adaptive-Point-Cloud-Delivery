import pandas as pd
from pandas.testing import assert_frame_equal

from train.tasks.semantic.modules.metrics import group_metrics

data = pd.DataFrame.from_dict({
    "a": [0.1, 0.2, 0.3, 0.4],
    "b": [0.5, 0.6, 0.7, 0.8],
})

expected = pd.DataFrame.from_dict({
    "a": [0.3, 0.7],
    "b": [1.1, 1.5],
})

expected_overhang = pd.DataFrame.from_dict({
    "a": [0.6, 0.4],
    "b": [1.8, 0.8],
})

def test_group_metrics_every_two_rows():
    actual = group_metrics(data=data, group_size=2)
    assert_frame_equal(actual, expected)

def test_group_metrics_no_grouping():
    actual = group_metrics(data=data, group_size=1)
    assert_frame_equal(actual, data)

def test_group_metrics_keep_remaining():
    actual = group_metrics(data=data, group_size=3)
    print(actual)
    assert_frame_equal(actual, expected_overhang)