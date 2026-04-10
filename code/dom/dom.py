# given the activations, this finds normalized directions for utility/safety
# it also shows a plot of their (safety/utility) cosine similarities across
# the different layers of the model

import numpy as np
import matplotlib as plt

def dom_cosine_similarities(base, group_a, group_b):
    '''
    base is what you compare from
    group_a is the direction of which we try to figure
    group_b is the other direction of which we try to figure
    it outputs an array containing cosine similarities of the 
    directions per each layer
    '''
    num_layers = base.shape[1]
    cosine_sims = []
    for layer in range(num_layers):
        base_mean = base[:, layer, :].mean(axis=0)
        a_mean = group_a[:, layer, :].mean(axis=0)
        b_mean = group_b[:, layer, :].mean(axis=0)

        direction_a = a_mean - base_mean
        direction_b = b_mean - base_mean

        direction_a /= np.linalg.norm(direction_a)
        direction_b /= np.linalg.norm(direction_b)
        cosine_sims.append(np.dot(direction_a, direction_b))
    return np.array(cosine_sims)

# TODO add permutation testing and the graph including confidence interval
