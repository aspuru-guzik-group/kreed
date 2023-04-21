import torch


def zeroed_com(M, x, orthogonal=False):
    coms = M.sum_pool(M.masses_normalized * x, broadcast=True)
    if orthogonal:
        norm = M.sum_pool(M.masses_normalized.square(), broadcast=True)
        shift = (coms * M.masses_normalized) / norm
    else:
        shift = coms
    return x - shift


@torch.no_grad()
def assert_zeroed_com(M, x):
    coms = M.sum_pool(M.masses_normalized * x, broadcast=True)
    error = coms.abs().max().item() / (x.abs().max().item() + 1e-5)
    assert error < 1e-4, error
