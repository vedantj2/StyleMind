from os import path
import torch
import torch.distributed as dist
import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

_src_path = path.join(path.dirname(path.abspath(__file__)), "src")
_USE_CPP_EXTENSION = True
_backend = None

try:
    _backend = load(name="inplace_abn",
                    extra_cflags=["-O3"],
                    sources=[path.join(_src_path, f) for f in [
                        "inplace_abn.cpp",
                        "inplace_abn_cpu.cpp",
                        "inplace_abn_cuda.cu",
                        "inplace_abn_cuda_half.cu"
                    ]],
                    extra_cuda_cflags=["--expt-extended-lambda"])
except Exception as e:
    print(f"Warning: Failed to load C++ extension for inplace_abn: {e}")
    print("Falling back to Python implementation (may be slower but works without C++ compiler)")
    _USE_CPP_EXTENSION = False
    _backend = None

# Activation names
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


def _act_forward(ctx, x):
    if not _USE_CPP_EXTENSION:
        # Python fallback - use functional API with inplace=True
        if ctx.activation == ACT_LEAKY_RELU:
            F.leaky_relu(x, negative_slope=ctx.slope, inplace=True)
        elif ctx.activation == ACT_ELU:
            F.elu(x, inplace=True)
        elif ctx.activation == ACT_NONE:
            pass
    else:
        if ctx.activation == ACT_LEAKY_RELU:
            _backend.leaky_relu_forward(x, ctx.slope)
        elif ctx.activation == ACT_ELU:
            _backend.elu_forward(x)
        elif ctx.activation == ACT_NONE:
            pass


def _act_backward(ctx, x, dx):
    if not _USE_CPP_EXTENSION:
        # Python fallback - compute gradients manually
        if ctx.activation == ACT_LEAKY_RELU:
            dx.mul_(torch.where(x > 0, torch.ones_like(x), torch.full_like(x, ctx.slope)))
        elif ctx.activation == ACT_ELU:
            dx.mul_(torch.where(x > 0, torch.ones_like(x), x + 1))
        elif ctx.activation == ACT_NONE:
            pass
    else:
        if ctx.activation == ACT_LEAKY_RELU:
            _backend.leaky_relu_backward(x, dx, ctx.slope)
        elif ctx.activation == ACT_ELU:
            _backend.elu_backward(x, dx)
        elif ctx.activation == ACT_NONE:
            pass


class InPlaceABN(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)

        if ctx.training:
            if _USE_CPP_EXTENSION:
                mean, var = _backend.mean_var(x)
            else:
                # Python fallback: compute mean and var
                mean = x.mean(dim=[0, 2, 3] if len(x.shape) == 4 else [0])
                var = x.var(dim=[0, 2, 3] if len(x.shape) == 4 else [0], unbiased=False)

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count / (count - 1))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        if _USE_CPP_EXTENSION:
            _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        else:
            # Python fallback: use PyTorch batch_norm
            # Reshape mean/var for broadcasting (use views, don't modify originals)
            if len(x.shape) == 4:
                mean_view = mean.view(1, -1, 1, 1)
                var_view = var.view(1, -1, 1, 1)
                if ctx.affine:
                    weight_view = weight.view(1, -1, 1, 1)
                    bias_view = bias.view(1, -1, 1, 1)
                else:
                    weight_view = None
                    bias_view = None
            else:
                mean_view = mean.view(1, -1)
                var_view = var.view(1, -1)
                if ctx.affine:
                    weight_view = weight.view(1, -1)
                    bias_view = bias.view(1, -1)
                else:
                    weight_view = None
                    bias_view = None
            
            # Batch normalization: (x - mean) / sqrt(var + eps)
            x.sub_(mean_view).div_(torch.sqrt(var_view + ctx.eps))
            if ctx.affine:
                x.mul_(weight_view).add_(bias_view)
        
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        ctx.mark_non_differentiable(running_mean, running_var)
        return x, running_mean, running_var

    @staticmethod
    @once_differentiable
    def backward(ctx, dz, _drunning_mean, _drunning_var):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            if _USE_CPP_EXTENSION:
                edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
            else:
                # Python fallback: compute edz and eydz
                if len(z.shape) == 4:
                    edz = dz.sum(dim=[0, 2, 3])
                    if ctx.affine:
                        y = (z - var.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + ctx.eps)
                        if ctx.affine:
                            y = y * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
                        eydz = (dz * y).sum(dim=[0, 2, 3])
                    else:
                        eydz = (dz * z).sum(dim=[0, 2, 3])
                else:
                    edz = dz.sum(dim=0)
                    if ctx.affine:
                        y = (z - var.view(1, -1)) / torch.sqrt(var.view(1, -1) + ctx.eps)
                        y = y * weight.view(1, -1) + bias.view(1, -1)
                        eydz = (dz * y).sum(dim=0)
                    else:
                        eydz = (dz * z).sum(dim=0)
        else:
            # TODO: implement simplified CUDA backward for inference mode
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))

        if _USE_CPP_EXTENSION:
            dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        else:
            # Python fallback: compute backward pass
            # This is a simplified version - for inference, gradients aren't needed
            if ctx.training:
                # Simplified backward for training
                if len(z.shape) == 4:
                    n = z.size(0) * z.size(2) * z.size(3)
                    dz_mean = dz.mean(dim=[0, 2, 3], keepdim=True)
                    dz_var = ((dz * z).mean(dim=[0, 2, 3], keepdim=True) - dz_mean * z.mean(dim=[0, 2, 3], keepdim=True))
                    dx = dz - dz_mean - dz_var * (z - z.mean(dim=[0, 2, 3], keepdim=True)) / (var.view(1, -1, 1, 1) + ctx.eps)
                    if ctx.affine:
                        dx = dx * weight.view(1, -1, 1, 1) / torch.sqrt(var.view(1, -1, 1, 1) + ctx.eps)
                else:
                    n = z.size(0)
                    dz_mean = dz.mean(dim=0, keepdim=True)
                    dz_var = ((dz * z).mean(dim=0, keepdim=True) - dz_mean * z.mean(dim=0, keepdim=True))
                    dx = dz - dz_mean - dz_var * (z - z.mean(dim=0, keepdim=True)) / (var.view(1, -1) + ctx.eps)
                    if ctx.affine:
                        dx = dx * weight.view(1, -1) / torch.sqrt(var.view(1, -1) + ctx.eps)
            else:
                # For inference, just return dz (gradients not needed)
                dx = dz
        # dweight = eydz * weight.sign() if ctx.affine else None
        dweight = eydz if ctx.affine else None
        if dweight is not None:
            dweight[weight < 0] *= -1
        dbias = edz if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01, equal_batches=True):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # count = _count_samples(x)
        batch_size = x.new_tensor([x.shape[0]], dtype=torch.long)

        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)

        if ctx.training:
            if _USE_CPP_EXTENSION:
                mean, var = _backend.mean_var(x)
            else:
                # Python fallback: compute mean and var
                mean = x.mean(dim=[0, 2, 3] if len(x.shape) == 4 else [0])
                var = x.var(dim=[0, 2, 3] if len(x.shape) == 4 else [0], unbiased=False)
            if ctx.world_size > 1:
                # get global batch size
                if equal_batches:
                    batch_size *= ctx.world_size
                else:
                    dist.all_reduce(batch_size, dist.ReduceOp.SUM)

                ctx.factor = x.shape[0] / float(batch_size.item())

                mean_all = mean.clone() * ctx.factor
                dist.all_reduce(mean_all, dist.ReduceOp.SUM)

                var_all = (var + (mean - mean_all) ** 2) * ctx.factor
                dist.all_reduce(var_all, dist.ReduceOp.SUM)

                mean = mean_all
                var = var_all

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            count = batch_size.item() * x.view(x.shape[0], x.shape[1], -1).shape[-1]
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * (float(count) / (count - 1)))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        if _USE_CPP_EXTENSION:
            _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        else:
            # Python fallback: use PyTorch batch_norm
            # Reshape mean/var for broadcasting (use views, don't modify originals)
            if len(x.shape) == 4:
                mean_view = mean.view(1, -1, 1, 1)
                var_view = var.view(1, -1, 1, 1)
                if ctx.affine:
                    weight_view = weight.view(1, -1, 1, 1)
                    bias_view = bias.view(1, -1, 1, 1)
                else:
                    weight_view = None
                    bias_view = None
            else:
                mean_view = mean.view(1, -1)
                var_view = var.view(1, -1)
                if ctx.affine:
                    weight_view = weight.view(1, -1)
                    bias_view = bias.view(1, -1)
                else:
                    weight_view = None
                    bias_view = None
            
            # Batch normalization: (x - mean) / sqrt(var + eps)
            x.sub_(mean_view).div_(torch.sqrt(var_view + ctx.eps))
            if ctx.affine:
                x.mul_(weight_view).add_(bias_view)
        
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        ctx.mark_non_differentiable(running_mean, running_var)
        return x, running_mean, running_var

    @staticmethod
    @once_differentiable
    def backward(ctx, dz, _drunning_mean, _drunning_var):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            if _USE_CPP_EXTENSION:
                edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
            else:
                # Python fallback: compute edz and eydz
                if len(z.shape) == 4:
                    edz = dz.sum(dim=[0, 2, 3])
                    if ctx.affine:
                        y = (z - var.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + ctx.eps)
                        y = y * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
                        eydz = (dz * y).sum(dim=[0, 2, 3])
                    else:
                        eydz = (dz * z).sum(dim=[0, 2, 3])
                else:
                    edz = dz.sum(dim=0)
                    if ctx.affine:
                        y = (z - var.view(1, -1)) / torch.sqrt(var.view(1, -1) + ctx.eps)
                        y = y * weight.view(1, -1) + bias.view(1, -1)
                        eydz = (dz * y).sum(dim=0)
                    else:
                        eydz = (dz * z).sum(dim=0)
            
            edz_local = edz.clone()
            eydz_local = eydz.clone()

            if ctx.world_size > 1:
                edz *= ctx.factor
                dist.all_reduce(edz, dist.ReduceOp.SUM)

                eydz *= ctx.factor
                dist.all_reduce(eydz, dist.ReduceOp.SUM)
        else:
            edz_local = edz = dz.new_zeros(dz.size(1))
            eydz_local = eydz = dz.new_zeros(dz.size(1))

        if _USE_CPP_EXTENSION:
            dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        else:
            # Python fallback: compute backward pass
            if ctx.training:
                # Simplified backward for training
                if len(z.shape) == 4:
                    dz_mean = dz.mean(dim=[0, 2, 3], keepdim=True)
                    dz_var = ((dz * z).mean(dim=[0, 2, 3], keepdim=True) - dz_mean * z.mean(dim=[0, 2, 3], keepdim=True))
                    dx = dz - dz_mean - dz_var * (z - z.mean(dim=[0, 2, 3], keepdim=True)) / (var.view(1, -1, 1, 1) + ctx.eps)
                    if ctx.affine:
                        dx = dx * weight.view(1, -1, 1, 1) / torch.sqrt(var.view(1, -1, 1, 1) + ctx.eps)
                else:
                    dz_mean = dz.mean(dim=0, keepdim=True)
                    dz_var = ((dz * z).mean(dim=0, keepdim=True) - dz_mean * z.mean(dim=0, keepdim=True))
                    dx = dz - dz_mean - dz_var * (z - z.mean(dim=0, keepdim=True)) / (var.view(1, -1) + ctx.eps)
                    if ctx.affine:
                        dx = dx * weight.view(1, -1) / torch.sqrt(var.view(1, -1) + ctx.eps)
            else:
                # For inference, just return dz (gradients not needed)
                dx = dz
        # dweight = eydz_local * weight.sign() if ctx.affine else None
        dweight = eydz_local if ctx.affine else None
        if dweight is not None:
            dweight[weight < 0] *= -1
        dbias = edz_local if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None


inplace_abn = InPlaceABN.apply
inplace_abn_sync = InPlaceABNSync.apply

__all__ = ["inplace_abn", "inplace_abn_sync", "ACT_RELU", "ACT_LEAKY_RELU", "ACT_ELU", "ACT_NONE"]
