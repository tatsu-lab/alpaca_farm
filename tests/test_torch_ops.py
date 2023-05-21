import torch

from alpaca_farm import torch_ops


def test_batch_select():
    input = torch.tensor(
        [
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ]
    )
    index = torch.tensor([[0, 1], [1, 0], [0, 0]])
    actual = torch_ops.batch_select(input, index)
    expected = torch.tensor([[0, 1], [0, 3], [6, 6]])
    assert actual.eq(expected).all()


def test_pad_sequence_from_left():
    sequences = [
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor(
            [
                3.0,
            ]
        ),
        torch.tensor(
            [
                6.0,
                7.0,
            ]
        ),
    ]
    expected = torch.tensor([[0.0, 1.0, 2.0], [-1.0, -1.0, 3.0], [-1.0, 6.0, 7.0]])
    actual = torch_ops.pad_sequence_from_left(sequences, batch_first=True, padding_value=-1)
    torch.testing.assert_close(actual, expected)
