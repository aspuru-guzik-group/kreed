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
    coms = M.sum_pool(M.masses_normalized * x)
    error = coms.abs().max().item() / (x.abs().max().item() + 1e-5)
    assert error < 1e-4, error


def dropout_unsigned_coords(M, prange):
    if isinstance(prange, float):
        p = torch.full([M.batch_size, 1], prange)
    else:
        assert len(prange) == 2
        assert 0 <= prange[0] <= prange[1]
        p = (prange[1] - prange[0]) * torch.rand([M.batch_size, 1]) + prange[0]
    p = p.to(M.coords)

    dropout_mask = (torch.rand_like(M.coords) < M.broadcast(p))
    cond_mask = M.cond_mask & (~dropout_mask)
    return M.replace(cond_mask=cond_mask, cond_labels=torch.where(cond_mask, M.cond_labels, 0.0))
