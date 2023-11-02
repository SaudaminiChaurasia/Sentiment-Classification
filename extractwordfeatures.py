#!/usr/bin/python

import random
import math
import re
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass

    words = re.findall(r'\w+', x) #Regular expression to find all
    
    # Dict to store feature vector
    feature_vector = {}
    
    # Create a feature for each unique word and count their occurrences
    for word in words:
        if word in feature_vector:
            feature_vector[word] += 1
        else:
            feature_vector[word] = 1
    
    return feature_vector
