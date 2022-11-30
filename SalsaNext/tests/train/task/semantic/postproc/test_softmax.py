import torch
from train.tasks.semantic.postproc.softmax import threshold_criticality, threshold_percentile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# threshold_criticality
threshold=0.5

def test_threshold_criticality_correct_shape():
    actual_shape = threshold_criticality(
        torch.softmax(torch.rand((1,3,10,20), device=device), dim=1),
        threshold).shape

    assert (10,20) == actual_shape

def test_threshold_criticality_prefer_lower_crit_is_crit1():
    actual_labels = threshold_criticality(
        torch.tensor([[
                [[0.40]],
                [[0.40]],
                [[0.20]]
        ]]),
        threshold)

    assert torch.equal(
        torch.tensor([
            [0],
        ]),
        actual_labels)

def test_threshold_criticality_under_threshold_is_crit2():
    actual_labels = threshold_criticality(
        torch.tensor([[
                [[0.40]],
                [[0.50]],
                [[0.10]]
        ]]),
        threshold)

    assert torch.equal(
        torch.tensor([
            [1],
        ]),
        actual_labels)

def test_threshold_criticality_under_threshold_is_crit3():
    actual_labels = threshold_criticality(
        torch.tensor([[
                [[0.40]],
                [[0.10]],
                [[0.50]]
        ]]),
        threshold)

    assert torch.equal(
        torch.tensor([
            [2],
        ]),
        actual_labels)

def test_threshold_criticality_equal_threshold_is_crit1():
    actual_labels = threshold_criticality(
        torch.tensor([[
                [[0.50]],
                [[0.40]],
                [[0.10]]
        ]]),
        threshold)

    assert torch.equal(
        torch.tensor([
            [0],
        ]),
        actual_labels)

def test_threshold_criticality_over_threshold_is_crit1():
    actual_labels = threshold_criticality(
        torch.tensor([[
                [[0.60]],
                [[0.30]],
                [[0.10]]
        ]]),
        threshold)

    assert torch.equal(
        torch.tensor([
            [0],
        ]),
        actual_labels)

# threshold_percentile
percentile=0.70

def test_threshold_percentile_unsorted_correct_shape():
    actual_shape = threshold_percentile(
        torch.softmax(torch.rand((1,3,10,20), device=device), dim=1),
        percentile,
        sort_softmax=False).shape
    assert (10,20) == actual_shape

def test_threshold_percentile_unsorted_equal_is_crit1():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.70]],
                [[0.20]],
                [[0.10]]
        ]], device=device),
        percentile,
        sort_softmax=False)

    assert torch.equal(
        torch.tensor([
            [0],
        ], device=device),
        actual_labels)

def test_threshold_percentile_unsorted_greater_than_is_crit1():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.75]],
                [[0.20]],
                [[0.5]]
        ]], device=device),
        percentile,
        sort_softmax=False)

    assert torch.equal(
        torch.tensor([
            [0],
        ], device=device),
        actual_labels)

def test_threshold_percentile_unsorted_smaller_than_is_crit2():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.60]],
                [[0.30]],
                [[0.10]]
        ]], device=device),
        percentile,
        sort_softmax=False)

    assert torch.equal(
        torch.tensor([
            [1],
        ], device=device),
        actual_labels)

def test_threshold_percentile_unsorted_smaller_than_is_crit3():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.10]],
                [[0.30]],
                [[0.60]]
        ]], device=device),
        percentile,
        sort_softmax=False)

    assert torch.equal(
        torch.tensor([
            [2],
        ], device=device),
        actual_labels)

def test_threshold_percentile_sorted_is_crit2():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.10]],
                [[0.60]],
                [[0.30]]
        ]], device=device),
        percentile,
        sort_softmax=True)

    assert torch.equal(
        torch.tensor([
            [2],
        ], device=device),
        actual_labels)

def test_threshold_percentile_unsorted_percentile_one_is_crit3():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.60]],
                [[0.30]],
                [[0.10]]
        ]], device=device),
        1.0,
        sort_softmax=False)

    assert torch.equal(
        torch.tensor([
            [2],
        ], device=device),
        actual_labels)

def test_threshold_percentile_unsorted_percentile_zero_is_crit1():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.10]],
                [[0.30]],
                [[0.60]]
        ]], device=device),
        0.0,
        sort_softmax=False)

    assert torch.equal(
        torch.tensor([
            [0],
        ], device=device),
        actual_labels)


def test_threshold_percentile_unsorted_ascending_crit1():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.40]],
                [[0.20]],
                [[0.40]]
        ]], device=device),
        0.3,
        sort_softmax=False,
        ascending=True)

    assert torch.equal(
        torch.tensor([
            [0],
        ], device=device),
        actual_labels)

def test_threshold_percentile_unsorted_descending_crit3():
    actual_labels = threshold_percentile(
        torch.tensor([[
                [[0.40]],
                [[0.20]],
                [[0.40]]
        ]], device=device),
        0.3,
        sort_softmax=False,
        ascending=False)

    assert torch.equal(
        torch.tensor([
            [2],
        ], device=device),
        actual_labels)   