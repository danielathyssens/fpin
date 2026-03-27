import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# TENSOR SPECFICATION FOR USING GPU
USE_GPU = torch.cuda.is_available()

CPU_TENSORS = {
    'long': torch.LongTensor,
    'float': torch.FloatTensor,
    'byte': torch.ByteTensor,
    'sparse.long': torch.sparse.LongTensor,
    'sparse.float': torch.sparse.FloatTensor,
    'sparse.byte': torch.sparse.ByteTensor,
}
GPU_TENSORS = {
    'long': torch.cuda.LongTensor,
    'float': torch.cuda.FloatTensor,
    'byte': torch.cuda.ByteTensor,
    'sparse.long': torch.cuda.sparse.LongTensor,
    'sparse.float': torch.cuda.sparse.FloatTensor,
    'sparse.byte': torch.cuda.sparse.ByteTensor,
}
TENSORS = GPU_TENSORS if USE_GPU else CPU_TENSORS


def move_to(instance, device, in_train=True):
    # print('instance', instance)
    if in_train:
        # try:
        instance_tuple = (entity.float() for entity in instance[:5])
        # except TypeError:
        #     instance_tuple = (entity.unsqueeze(0) for entity in instance[:5])
    else:
        # try:
        instance_tuple = (torch.tensor(entity).unsqueeze(0).float() for entity in instance[:5])
        # except TypeError:
        #     instance_tuple = (entity.unsqueeze(0) for entity in instance[:5])
    fleet_b, depot_b, custom_b, demand_b, dists_b = instance_tuple
    return to_variable(fleet_b, depot_b, custom_b, demand_b, dists_b, device=device)

#             # Transfer to GPU
#             if torch.cuda.device_count() > 1:
#                 depot_batch, custom_batch, fleet_batch, dists_batch, demand_batch = to_variable(depot_batch, custom_batch, fleet_batch, dists_batch, demand_batch, device='cuda:0')
#             else:
#                 depot_batch, custom_batch, fleet_batch, dists_batch, demand_batch = to_variable(depot_batch, custom_batch, fleet_batch, dists_batch, demand_batch, device=device)


# ref: https://stackoverflow.com/questions/56880166/how-to-multiply-a-dense-matrix-by-a-sparse-matrix-element-wise-in-pytorch
# def sparse_dense_mul(s, d):
#  i = s._indices()
#  v = s._values()
#  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
#  return torch.sparse.FloatTensor(i, v * dv, s.size())

# ref: https://stackoverflow.com/questions/56880166/how-to-multiply-a-dense-matrix-by-a-sparse-matrix-element-wise-in-pytorch
# def sparse_dense_mul(s, d):
#     i = s._indices()
#     v = s._values()
#     dv = d[i[0, :], i[1, :], i[2, :], i[3, :]]  # get values from relevant entries of dense matrix
#     return torch.sparse.FloatTensor(i, v * dv, s.size())

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    # Move dense tensor to same device as sparse values
    d = d.to(v.device)
    # Extract matching dense values
    dv = d[i[0, :], i[1, :], i[2, :], i[3, :]]
    return torch.sparse_coo_tensor(i, v * dv, s.size(), dtype=torch.float, device=v.device)


# def sparse_dense_mul_loss(s, d):
#
#     i = s._indices()
#     v = s._values()
#     dv = d[i[0, :], i[1, :], i[2, :], i[3, :], i[4, :], i[5, :]]  # get values from relevant entries of dense matrix
#     # return torch.sparse.FloatTensor(i, v * dv, s.size())
#     return torch.sparse_coo_tensor(i, v * dv, s.size(), dtype=torch.float) # , device=

def sparse_dense_mul_loss(s, d):
    # print('d.device', d.device)
    # print('s.device', s.device)

    i = s._indices()
    v = s._values()
    assert i.shape[0] == 6, f"Expected 6D indices, got {i.shape[0]}D"
    # Move dense tensor to same device as sparse values
    d = d.to(v.device).contiguous()
    # print('AFTER')
    # print('d.device', d.device)
    # print('s.device', s.device)
    # print('i.device', i.device)
    # print('v.device', v.device)
    assert d.dtype == torch.float32, f"d has wrong dtype: {d.dtype}"
    assert v.dtype == torch.float32 or v.dtype == torch.uint8, f"v has wrong dtype: {v.dtype}"
    assert d.device == v.device, "Device mismatch between dense and sparse"
    assert d.is_contiguous(), "Dense tensor not contiguous"
    for dim in range(i.shape[0]):
        # print(f"i[{dim}].min():", i[dim].min().item(), " i[{dim}].max():", i[dim].max().item(), "  vs d.size({dim}):",
        #       d.size(dim))
        assert i[dim].min() >= 0
        assert i[dim].max() < d.size(dim)
    assert s.size() == d.size(), f"Shape mismatch: s={s.size()}, d={d.size()}"
    assert i.shape[0] == d.dim(), f"Expected index dim to match dense dim: {i.shape[0]} vs {d.dim()}"
    # Extract matching dense values
    # dv = d[i[0, :], i[1, :], i[2, :], i[3, :], i[4, :], i[5, :]]
    try:
        dv = d[i[0, :], i[1, :], i[2, :], i[3, :], i[4, :], i[5, :]]
    except IndexError:
        print("Index out of bounds! Dumping stats:")
        for dim in range(i.shape[0]):
            print(f"i[{dim}].max(): {i[dim].max()} vs d.size({dim}) = {d.size(dim)}")
    return torch.sparse_coo_tensor(i, v * dv, s.size(), dtype=torch.float, device=v.device)


def ones(*sizes, type=None):
    if type is not None:
        return TENSORS[type](*sizes).fill_(1)
    elif USE_GPU:
        # noinspection PyArgumentList
        return torch.cuda.FloatTensor(*sizes).fill_(1)
    else:
        return torch.ones(*sizes)


def is_sorted(tensor):
    # arr:  Tensor(length)
    # noinspection PyUnresolvedReferences
    return (tensor[:-1] <= tensor[1:]).all()


def to_variable(*tensors, device="cuda", non_blocking=False, pin_memory=False, volatile=False):
    result = []
    for tensor in tensors:
        if pin_memory:
            tensor = tensor.pin_memory()
        if not isinstance(tensor, Variable):
            tensor = Variable(tensor, volatile=volatile)
        if USE_GPU:
            # tensor = tensor.cuda(non_blocking=non_blocking)
            tensor = tensor.to(device)
        result.append(tensor)

    if len(result) == 1:
        return result[0]
    else:
        return result


def get_data(tensor):
    if isinstance(tensor, Variable):
        return tensor.data
    return tensor


class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        elif len(args) > 1:
            return args


def normalize_dims(x, dims):
    sums = x
    for dim in dims:
        sums = sums.sum(dim=dim, keepdim=True)
    sums = sums.expand_as(x)
    return x / sums


def zeros(*sizes, type=None):
    if type is not None:
        return TENSORS[type](*sizes).fill_(0)
    elif USE_GPU:
        # noinspection PyArgumentList
        return torch.cuda.FloatTensor(*sizes).fill_(0)
    else:
        return torch.zeros(*sizes)


# REF: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
from typing import Union


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def strip_main_diagonal(x):
    # x:    Float(dim_0 x dim_1 x ... dim_(k-2) x n x n)
    # res:  Float(dim_0 x dim_1 x ... dim_(k-2) x n x (n-1))

    # removes the main diagonal from x, e.g:
    # From:  0 1 2
    #        3 4 5
    #        6 7 8
    # To:    1 2
    #        3 5
    #        6 7
    # by first changing x such that the main diagonal is on the first column (while discarding the last item):
    #        0 1 2 3
    #        4 5 6 7
    # and then stripping it and reshaping the tensor

    assert x.dim() >= 2, 'can only strip main diagonal of tensor of two or more dimensions!'
    assert x.size(-1) == x.size(-2), 'strip_main_diagonal currently supports striping diagonals of square dimensions' \
                                     ' only!'
    assert x.size(-1) > 1, 'cannot strip main diagonal if the dimension size is only 1!'

    base_size = x.size()[:-2]
    n = x.size(-1)

    # merge the last two dimensions
    new_x = x.contiguous().view(*base_size, n * n)

    # remove the last item on the diagonal
    new_x = new_x.transpose(0, x.dim() - 2)[:-1].transpose(0, x.dim() - 2)

    # view x such that the first column is the diagonal
    new_x = new_x.contiguous().view(*base_size, n - 1, n + 1)

    # remove the first column
    new_x = new_x.transpose(0, x.dim() - 1)[1:].transpose(0, x.dim() - 1)

    # reshape to result
    result = new_x.contiguous().view(*base_size, n, n - 1)

    return result


def chunk_at(tensor, dim=0, squeeze=True):
    if squeeze:
        return (t.squeeze(dim) for t in tensor.chunk(tensor.size(dim), dim=dim))
    else:
        return tensor.chunk(tensor.size(dim), dim=dim)


def sparse_transpose_dim2_dim3(sparse_tensor):
    """
    Transpose a sparse 4D tensor along dims 2 and 3, assuming shape (B, M, N, N).
    Returns a new sparse tensor with the transposed dimensions.
    """
    assert sparse_tensor.is_sparse, "Tensor must be sparse"
    assert sparse_tensor.dim() == 4, "Only supports 4D sparse tensors (B, M, N, N)"

    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    size = sparse_tensor.size()

    # Swap dim 2 and 3 in the indices
    indices_swapped = indices.clone()
    indices_swapped[2], indices_swapped[3] = indices[3], indices[2]

    return torch.sparse_coo_tensor(indices_swapped, values, size=size, device=sparse_tensor.device)


def sparse_stack(tensors, dim=0):
    """
    Stack a list of sparse COO tensors along a new dimension `dim`.
    Only supports COO format. Returns a sparse tensor.
    """
    assert all(t.is_sparse for t in tensors), "All tensors must be sparse"
    assert len(tensors) > 0, "Need at least one tensor"

    device = tensors[0].device
    dtype = tensors[0].dtype
    sizes = [t.size() for t in tensors]
    assert all(s == sizes[0] for s in sizes), "All sparse tensors must have same shape"

    base_shape = sizes[0]
    stacked_shape = list(base_shape)
    stacked_shape.insert(dim, len(tensors))  # add new dim

    new_indices = []
    new_values = []

    for i, t in enumerate(tensors):
        idx = t._indices()
        val = t._values()
        # Expand indices to one more dim
        # Insert the new dim at the correct position with fill value i
        prefix = idx[:dim]
        suffix = idx[dim:]
        new_dim = torch.full((1, idx.shape[1]), i, dtype=torch.long, device=idx.device)
        new_idx = torch.cat([prefix, new_dim, suffix], dim=0)
        new_indices.append(new_idx)
        new_values.append(val)

    final_indices = torch.cat(new_indices, dim=1)
    final_values = torch.cat(new_values)

    return torch.sparse_coo_tensor(final_indices, final_values, size=tuple(stacked_shape), dtype=dtype, device=device)


def safe_gather(tensor, dim, index):
    assert index.max() < tensor.size(dim), f"GATHER ERROR: max index {index.max()} exceeds dim {tensor.size(dim)}"
    assert index.min() >= 0, "GATHER ERROR: negative index!"
    return tensor.gather(dim, index)