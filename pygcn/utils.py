import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder


def load_data(path="./data/cora/", dataset="cora"):
    """
    Load citation network dataset (cora only for now)
    """

    print('Loading {} dataset...'.format(dataset))

    # read the whole data using str first due to the format of the labels
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    
    # use `Sparse Row format` to gerenate a sparse mat.
    # skip the id & the label to read the word vecs.
    X = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # read the labels and index them
    label_encoder = LabelEncoder()
    labels_scalar = label_encoder.fit_transform(idx_features_labels[:, -1])
    num_classes = len(label_encoder.classes_)

    # build graph
    # read all the nodes
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}         # map paper-id to index


    # read all the edges
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    
    # use the index number instead of the original paper-id (TODO WHY?)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    
    # set the adjacency mat.
    # It's more convenient to use `COOrdinate format` to gerenate a sparse mat.
    # A[i[k], j[k]] = data[k]
    num_nodes = labels_scalar.shape[0]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes),         # N x N
                        dtype=np.float32).tocsr()       # convert to csr format for better pf

    # generate symmetric adjacency matrix by performing point-wise multiplication
    # adj.T  adj    adj.T > adj     shall we?
    #   0     0         F               N
    #   0     1         F               N
    #   1     0         T               Y
    #   1     1         F               N
    # if A-ij > 1, then b = a + b - a & "adj = adj.T".
    # `multiply(adj.T > adj)` indicates elements need to be modified.
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)     

    # normalize word vecs
    X = normalize_features(X)

    # add self-connection for each node
    adj_hat = adj + sp.eye(adj.shape[0], format="csr")

    # perform random walk normalization
    # DA = random_walk_normalization(adj + sp.eye(adj.shape[0]))

    # perform symmetric normalization
    DAD = symmetric_normalization(adj_hat)

    idx_train = torch.LongTensor(range(140))
    idx_val = torch.LongTensor(range(200, 500))
    idx_test = torch.LongTensor(range(500, 1500))

    X = torch.FloatTensor(np.array(X.todense()))
    labels_scalar = torch.LongTensor(labels_scalar)       # get scalar label from col-idx
    
    DAD = sparse_mx_to_torch_sparse_tensor(DAD)
    
    return DAD, X, labels_scalar, num_classes, idx_train, idx_val, idx_test


def normalize_features(X):
    """
    Row-normalize sparse matrix.
    """

    rowsum = np.array(X.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.                 # rowsum == 0
    r_mat_inv = sp.diags(r_inv)
    X = r_mat_inv.dot(X)                        # row-wise
    return X

def random_walk_normalization(adj):
    """
    Row-normalize sparse matrix.
    It behaves exactly the same as the func `normalize_features`.
    """

    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.                   # rowsum == 0
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)                      # row-wise
    return adj

def symmetric_normalization(adj : sp.csr_matrix):

    # calculate degrees for each node
    degrees = adj.sum(axis=0)

    # generate sparse matrix D using the degrees of nodes.
    D = sp.dia_matrix((degrees, 0), shape=adj.shape).tocsr()

    # derive the sym-norm matrix
    D_pow = D.power(-0.5)

    return D_pow.dot(adj).dot(D_pow)            # DAD


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """

    # make sure it's coo format first
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    # TODO probe the variable: indices
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )

    # get all the values corresponding to the indices
    values = torch.from_numpy(sparse_mx.data)

    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):

    # use scalar labels
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


# unit testing
if __name__ == "__main__":

    adj = np.array([[0, 1, 1, 1],
                    [1, 0, 1, 0],
                    [1, 1, 0, 0],
                    [1, 0, 0, 0]])
    
    adj = sp.csr_matrix(adj, dtype=np.float32)
    
    adj_hat = adj + sp.eye(adj.shape[0], format="csr")

    DAD = symmetric_normalization(adj_hat)

    print(DAD.todense())

