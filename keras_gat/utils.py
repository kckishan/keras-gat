from __future__ import print_function

import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    FILE_PATH = os.path.abspath(__file__)
    DIR_PATH = os.path.dirname(FILE_PATH)
    DATA_PATH = os.path.join(DIR_PATH, 'data/')

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATA_PATH, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


### Added utility functions for protein function classification

def _load_network(filename, num_genes, mtrx='adj'):
    print ("### Loading [%s]..." % (filename))
    if mtrx == 'adj':
        i, j, val = np.loadtxt(filename).T
        A = coo_matrix((val, (i.astype(int)-1, j.astype(int)-1)), shape=(num_genes, num_genes))
        A = A.todense()
        A = np.squeeze(np.asarray(A))
        if A.min() < 0:
            print ("### Negative entries in the matrix are not allowed!")
            A[A < 0] = 0
            print ("### Matrix converted to nonnegative matrix.")
            print()
        if (A.T == A).all():
            pass
        else:
            print ("### Matrix not symmetric!")
            A = A + A.T
            print ("### Matrix converted to symmetric.")
    else:
        print ("### Wrong mtrx type. Possible: {'adj', 'inc'}")
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=1) == 0)
    np.fill_diagonal(A, 0)

    return A

def load_protein_function(org):
    """Load data."""
    FILE_PATH = os.path.abspath(__file__)
    DIR_PATH = os.path.dirname(FILE_PATH)
    DATA_PATH = os.path.join(DIR_PATH, 'data/')

    # get number of nodes N to create N * N adjacency matrix
    import pandas as pd
    geneids = pd.read_csv(DATA_PATH + '/'+org+'/' + org + '_string_genes.txt', sep="\t", header=None)
    num_genes = geneids.shape[0]

    file_name = DATA_PATH + '/'+org+'/' + org + '_string_neighborhood_adjacency.txt'
    labels = sio.loadmat(DATA_PATH + '/'+org+'/' + org + '_annotations.mat')
    adj = _load_network(file_name, num_genes)

    # Assign the adjacency matrix as features
    features = sp.csr_matrix(adj).tolil()
    # Define identity matrix as features
#    features = sp.csr_matrix(np.eye(adj)).tolil()

    # Binarize the adjacency matrix
    A = binarize(adj, 0)


    labels = labels['level1']

    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(idx, test_size=0.4, random_state=0)
    idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=0)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask