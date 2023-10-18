from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN


def train(epoch):
    t = time.time()

    model.train()
    optimizer.zero_grad()

    output = model(X, DAD)

    # use training samples only to train the model
    # Note: F.nll_loss use scalar labels, not one-hot labels!
    loss_train = F.nll_loss(output[idx_train], labels_scalar[idx_train])
    loss_train.backward()
    optimizer.step()

    acc_train = accuracy(output[idx_train], labels_scalar[idx_train])

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(X, DAD)

    loss_val = F.nll_loss(output[idx_val], labels_scalar[idx_val])
    acc_val = accuracy(output[idx_val], labels_scalar[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():

    model.eval()

    # get results on test set & cal loss
    output = model(X, DAD)
    loss_test = F.nll_loss(output[idx_test], labels_scalar[idx_test])

    acc_test = accuracy(output[idx_test], labels_scalar[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Skip validation during training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    DAD, X, labels_scalar, num_classes, idx_train, idx_val, idx_test = load_data()

    # Model and optimizer
    model = GCN(nfeat=X.shape[1],
                nhid=args.hidden,
                nclass=num_classes,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, 
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        X = X.cuda()
        DAD = DAD.cuda()
        labels_scalar = labels_scalar.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
    # Train model
    t_total = time.time()

    for epoch in range(args.epochs):
        train(epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

