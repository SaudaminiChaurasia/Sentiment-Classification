#!/usr/bin/python

import random
import math
import re
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        x = x.replace(" ", "").replace("\t", "")

        # Initialize an empty feature vector
        feature_vector = {}
    
        # Generate n-grams and count their occurrences
        for i in range(len(x) - n + 1):
            ngram = x[i:i + n]
        
            if ngram in feature_vector:
                feature_vector[ngram] += 1
            else:
                feature_vector[ngram] = 1

        return feature_vector

    return extract
