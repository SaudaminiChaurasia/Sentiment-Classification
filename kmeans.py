#!/usr/bin/python

import random
import math
import re
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


def compute_distance(example1, example2):
    # Compute the Euclidean distance between two sparse vectors (string-to-float dictionaries)
    distance = 0.0
    for key in example1:
        if key in example2:
            distance += (example1[key] - example2[key])**2
        else:
            distance += example1[key]**2
    for key in example2:
        if key not in example1:
            distance += example2[key]**2
    return math.sqrt(distance)

def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    
    # Initialize cluster centers to random elements from examples
    random.seed(0)
    centers = random.sample(examples, K)
    assignments = [0] * len(examples)
    
    for epoch in range(maxEpochs):
        # Assign each example to the nearest center
        new_assignments = [0] * len(examples)
        for i, example in enumerate(examples):
            min_distance = float('inf')
            for j, center in enumerate(centers):
                distance = compute_distance(center, example)
                if distance < min_distance:
                    min_distance = distance
                    new_assignments[i] = j
        
        # Update cluster centers based on the new assignments
        new_centers = [{} for _ in range(K)]
        cluster_sizes = [0] * K
        for i, assignment in enumerate(new_assignments):
            for key, value in examples[i].items():
                new_centers[assignment][key] = new_centers[assignment].get(key, 0) + value
            cluster_sizes[assignment] += 1
        
        for j in range(K):
            for key in new_centers[j]:
                new_centers[j][key] /= cluster_sizes[j]
        
        # Check for convergence
        if new_assignments == assignments:
            break
        
        assignments = new_assignments
        centers = new_centers
    
    # Compute the final reconstruction loss
    loss = 0.0
    for i, example in enumerate(examples):
        loss += compute_distance(centers[assignments[i]], example)
    
    return centers, assignments, loss
