import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        dropout: probability to conduct dropout strategy.

        The letter "n" stands for "the number of ... "
        """

        super(GCN, self).__init__()

        # Define two gcn filters.
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        # Use the dropout strategy to do regularization.
        self.dropout = dropout

    def forward(self, x, adj):

        # Activate the output of the first gcn filter.
        x = F.relu(self.gc1(x, adj))

        # Perform dropout when we are training the model.
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)

        # F.log_softmax is mathematically equivalent to log(softmax(x)), 
        # but it computes more stably than that.
        return F.log_softmax(x, dim=1)
    
