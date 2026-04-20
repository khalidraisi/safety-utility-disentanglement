# this file has the implementations of ActSVD and contrastive SVD
# Wei et al have 2 versions, they call them ActSVD (Top) as well as ActSVD (with orthogonal projection)
# They describe ActSVD (Top) as it gives you the top r principal components forming the r-dimensional subspace,
# and "top" refers to the ranking of singular values and their respective principal components
# They describe ActSVD (Orthogonal projection) as it gives you the part of the safety subspace that does not overlap with utility
# Both are useful for our purposes (with a bit of tweaking), they are described below

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_topr_sv(activations, r):
    '''
    this function computes the top r singualar vectors
    takes the activations and r, 
    returns a matrix with orthonormal columns, those columns are 
    also the top r singular vectors
    '''
    X = activations.T 
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return U[:, :r]


# ActSVD (Top)
def actsvd_top(activations, r):
    '''
    process the activations and call the function for top r singular values
    takes activations and r for all activation layers
    returns a list of matrices with orthonormal columns (See compute_topr_sv)
    '''
    num_layers = activations.shape[1]
    return [compute_topr_sv(activations[:, i, :], r) for i in range(num_layers)]


#ActSVD (Orthogonal projection)
def actsvd_op(subspace_A, subspace_B):
    '''
    removes the part of subspace_A that overlaps with subspace_B from subspace_A
    returning the part of subspace_A that does not overlap with subspace_B
    and the norms of the survived subspace basis
    '''
    num_layers = len(subspace_A)
    isolated = []
    survival_norms = []

    for i in range(num_layers):
        U_A = subspace_A[i]
        U_B = subspace_B[i]
        # projection matrix onto subspace_B
        Pi_B = U_B @ U_B.T

        # subtract A component inside B
        U_A_overlap = Pi_B @ U_A
        U_A_isolated = U_A - U_A_overlap
        
        #added this to know how much of subspace_A survives the cut
        #a norm closer to 0 means that subspace_A was almost completely overlapping
        #a norm closer to 1 means that subspace_A had almost no overlap
        #i think without this we'd get some random strange resutls
        #this is just a check
        col_norms = np.linalg.norm(U_A_isolated, axis=0)
        survival_norms.append(col_norms)

        #orthonormalize
        Q, R = np.linalg.qr(U_A_isolated)
        isolated.append(Q[:, :U_A.shape[1]])

    return isolated, survival_norms    

# contrastive SVD
def contrastive_svd(tgt_acts, baseline_acts, r):
    '''
    takes the difference between 2 activations from paired prompts
    and then run SVD on that to find the top r directions that distinguishes 
    between the 2 prompt sets
    returns a list of matrices with orthonormal columns
    '''
    assert tgt_acts.shape == baseline_acts.shape, \
        "target and baseline are different shapes"

    num_layers = tgt_acts.shape[1]
    subspaces = []

    for i in range(num_layers):
        differences = tgt_acts[:, i, :] - baseline_acts[:, i, :]
        subspaces.append(compute_topr_sv(differences, r))

    return subspaces

# add analyzing part here maybe
#plot of survival norms
def survival_plot(survival_norms):
    '''
    Plots per-column survival norms from actsvd_op across layers.
    Each dot is one basis direction of the isolated subspace; y-axis is log-scale.
    '''
    num_layers = len(survival_norms)

    plt.figure(figsize=(12, 6))

    rng = np.random.default_rng(0)
    for layer_idx, norms in enumerate(survival_norms):
        xs = layer_idx + rng.uniform(-0.15, 0.15, size=len(norms))
        ys = np.clip(norms, 1e-18, None)  # avoid -inf on log scale
        plt.scatter(xs, ys, s=25, alpha=0.6, edgecolor='none', color='blue',
                    label="Column survival norm" if layer_idx == 0 else None)

    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.yscale("log")
    plt.xlabel("Layer number (0 is embedding layer)")
    plt.ylabel("Survival norm  ||(I - Π_B) u||")
    plt.title("Survival norms of isolated subspace per layer")
    plt.xticks(range(0, num_layers, 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()