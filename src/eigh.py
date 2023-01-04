import torch
import torch.linalg as LA


class StablePowerIteration(torch.autograd.Function):

    @staticmethod
    def forward(ctx, M, v, n_iter=19):
        ctx.n_iter = n_iter
        ctx.save_for_backward(M, v)
        return v

    @staticmethod
    def backward(ctx, grad_output):
        M, v = ctx.saved_tensors  # (B 3 3) (B 3 1)
        dL_dv = grad_output
        I = torch.eye(M.shape[-1]).to(M).unsqueeze(0).repeat(M.shape[0], 1, 1)  # (B 3 3)

        num = I - torch.bmm(v, torch.transpose(v, -1, -2))
        denom = LA.norm(torch.bmm(M, v), dim=(1, 2), keepdim=True).clamp(min=1e-5)
        ak = num / denom

        term1 = ak
        q = M / denom
        for i in range(1, ctx.n_iter + 1):
            ak = torch.bmm(q, ak)
            term1 += ak
        dL_dM = torch.bmm(torch.bmm(term1, dL_dv), torch.transpose(v, -1, -2))

        return dL_dM, ak, None
