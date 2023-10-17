import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):

        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Define layer weights for this GCN filter to change the # of input channels.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # Define a selectable bias for each node.
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            # The parameter can be accessed from this module using the given name.
            self.register_parameter('bias', None)

        # Note 因为激活函数不属于“图卷积操作”，所以没有加入到这一层的抽象中
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        We initialize weights using the initialization described in Glorot & Bengio (2010).
        """

        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        adj: the `DAD` part
        input: the feature matrix X
        """
        
        # get the MLP part
        # torch.mm: Performs a matrix multiplication of the matrices input and mat2.
        support = torch.mm(input, self.weight)

        # The adjacency matrix is sparse.
        # So we can use sparse matrix multiplication algo (spmm) to reduce time complexity.
        output = torch.spmm(adj, support)


        # perform row-wise broadcast & **introduce the same bias to each node**
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

