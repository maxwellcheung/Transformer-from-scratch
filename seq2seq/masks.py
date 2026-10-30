import torch

def make_padding_mask(tokens, pad_id):
    if tokens.dim() != 2:
        raise ValueError(
            f'expected tokens with 2 dimm, got {tokens.dim()}'
        )
    return tokens.eq(pad_id).unsqueeze(1).unsqueeze(2)

def  make_causal_mask(length, device):
    if length <= 0:
        raise ValueError(f'length must be positive, got {length}')

    mask = torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device,),
        diagonal=1,
    )

    return mask.unsqueeze(0).unsqueeze(0)