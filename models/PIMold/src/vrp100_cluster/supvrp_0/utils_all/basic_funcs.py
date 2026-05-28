
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

# ref: https://stackoverflow.com/questions/56880166/how-to-multiply-a-dense-matrix-by-a-sparse-matrix-element-wise-in-pytorch
# def sparse_dense_mul(s, d):
#  i = s._indices()
#  v = s._values()
#  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
#  return torch.sparse.FloatTensor(i, v * dv, s.size())

# ref: https://stackoverflow.com/questions/56880166/how-to-multiply-a-dense-matrix-by-a-sparse-matrix-element-wise-in-pytorch
def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:], i[2,:], i[3,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())

def sparse_dense_mul_loss(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:], i[2,:], i[3,:], i[4,:], i[5,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())

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

def to_variable(*tensors, non_blocking=False, pin_memory=False, volatile=False):
    result = []
    for tensor in tensors:
        if pin_memory:
            tensor = tensor.pin_memory()
        if not isinstance(tensor, Variable):
            tensor = Variable(tensor, volatile=volatile)
        if USE_GPU:
            tensor = tensor.cuda(non_blocking=non_blocking)
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