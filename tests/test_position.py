import pytest
import torch

def test_position_supports_odd_model_width():
    from seq2seq.position import PositionalEncoding

    module = PositionalEncoding(
        d_model=5,
        max_len=4,
        dropout=0.0,
    )

    x = torch.zeros(2, 4, 5)
    y = module(x)

    assert y.shape == x.shape
    torch.testing.assert_close(y[0], y[1])

    torch.testing.assert_close(
        y[0,0],
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
    )

def test_position_rejects_sequence_longer_than_table():
    from seq2seq.position import PositionalEncoding

    module = PositionalEncoding(d_model=4, max_len=3)
    with pytest.raises(
        ValueError,
        match='sequence length 4.*max_len 3',
    ):
        module(torch.zeros(1,4,4))

def test_position_rejects_wrong_model_width():
    from seq2seq.position import PositionalEncoding

    module = PositionalEncoding(d_model=4, max_len=3)
    with pytest.raises(
        ValueError,
        match='expected.*4.*got.*5',
    ):
        module(torch.zeros(2,3,5))

def test_position_table_is_buffer_not_parameter():
    from seq2seq.position import PositionalEncoding
    module = PositionalEncoding(d_model=4, max_len=3)

    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    assert 'pe' not in parameters
    assert 'pe' in buffers

def test_position_preserves_input_dtype():
    from seq2seq.position import PositionalEncoding

    module = PositionalEncoding(
        d_model=4,
        max_len=3,
        dropout=0.0,
    )

    x = torch.zeros(1,3,4,dtype=torch.float16)
    y = module(x)

    assert y.dtype == torch.float16

@pytest.mark.parametrize(
    'd_model, max_len, message',
    [
        (0, 4, 'd_model.*0'),
        (4, 0, 'max_len.*0'),
    ],
)

def test_position_rejects_non_positive_configuraion(
    d_model,
    max_len,
    message,
):
    from seq2seq.position import PositionalEncoding

    with pytest.raises(ValueError,match=message):
        PositionalEncoding(d_model=d_model, max_len=max_len,)