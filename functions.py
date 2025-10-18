#!/env/bin/activate

from typing import Literal
import numpy as np
import pandas as pd

DATATYPE = Literal["PB", "DS"]


def entropy(
    dataset: np.ndarray | pd.Series | list, base: int = 10, datatype: DATATYPE = "PB"
) -> float:
    """

     @brief Computes the entropy of a dataset for a given logarithmic base and data type.

     This function calculates the entropy (information content) of a dataset using the
     Shannon entropy formula:
     \f[
         H(X) = -\sum_i p_i \log_b(p_i)
     \f]
     where \f$p_i\f$ is the probability of occurrence of each unique element in the dataset,
     and \f$b\f$ is the logarithm base (e.g., 2 for bits, 10 for decimal entropy, etc.).

     @param dataset
         The input data to compute entropy from. Can be a NumPy array, Pandas Series, or list.

     @param base
         The base of the logarithm to use for entropy calculation.
         Defaults to 10.

     @param datatype
         Specifies how to interpret the input dataset:
           - `'PB'`: Probability-based input (dataset already contains probability values).
           - `'DS'`: Discrete sample input (dataset contains raw discrete values;
                     probabilities are computed internally).

     @return float
         The computed entropy value.

     @throws AssertionError
         If an invalid datatype is provided or if the dataset type is unsupported.

     @note
         - Zero values are ignored in logarithmic calculations (since log(0) is undefined).
         - Entropy is always non-negative.
         - The function automatically converts lists and Pandas Series into NumPy arrays.

     @example
         >>> entropy([0.2, 0.5, 0.3], base=2, datatype='PB')
         1.4854752972273344

         >>> entropy([1, 1, 2, 2, 3, 3, 3], base=2, datatype='DS')
         1.5566567074628228
    /
    """

    VALID_DATATYPES = DATATYPE.__args__

    log_base = lambda x, b: (np.log(x) / np.log(b)) if x != 0 else 0
    cal = lambda x: x * log_base(x, base)

    assert (
        datatype in VALID_DATATYPES
    ), f"Invalid datatype provided: '{datatype}'. Must be one of {VALID_DATATYPES}."
    assert isinstance(
        dataset, (pd.Series, np.ndarray, list)
    ), f"Invalid datatype passed: '{type(dataset)}'. Must provide np.ndarray, pd.Series, or list."

    # Convert input to numpy array
    if isinstance(dataset, pd.Series):
        dataset = dataset.to_numpy()
    elif isinstance(dataset, list):
        dataset = np.array(dataset)

    if len(dataset) == 0:
        return 0.0

    # If data is discrete, compute probabilities
    if datatype == "DS":
        unique = np.unique(dataset)
        dataset = np.array([np.sum(dataset == i) / len(dataset) for i in unique])

    return -np.sum(list(map(cal, dataset)))


def joined_entropy(
    dataset_a: np.ndarray | pd.Series | list,
    dataset_b: np.ndarray | pd.Series | list,
    base: int = 10,
    datatype: DATATYPE = "PB",
) -> np.float64:

    pass


# Complete one for mutual information
