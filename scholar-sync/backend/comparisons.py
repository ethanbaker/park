import numpy as np

def text_comparison(a: np.ndarray[any], b: np.ndarray[any]) -> float:
    """
    Compare two vectorsMap all words in a paragraph to their corresponding
    word vector. Take the sum of all these vectors together as the paragraph
    vector. This vector can be used to calculate the distance between other
    paragraph vector representations for other students. Vector representations
    are saved as a unit vector in N-dimensional space.

    Vectors A and B are compared between two different students. Compute the 
    “closeness” score by taking the dot products between the two vectors. Two 
    vectors are closer for more positive values, further for more negative 
    values, and zero correlates to the vectors having no impact on one another. 

    :param a: The first vector to compare
    :param b: The second vector to compare
    :return: float representation of how close the two values are (1 = more similar, -1 = less similar)
    """
    a_hat = a / sum([v**2 for v in a])**0.5
    b_hat = b / sum([v**2 for v in b])**0.5

    return float(np.dot(a_hat, b_hat))

def enum_comparison(n: int):
    """
    Have value A and value B that are represented as numerical values. This can
    be from an Enum or as a raw number. 
    
    The number range is N (max_val - min_val = N).

    Values A and B are compared using their innate differences. The wider the 
    difference, the more negative the result. The smaller the difference, the 
    more positive the result. A score of zero correlates to no impact. 

    :param a: The first value to compare
    :param b: The second value to compare
    :param n: The range of the values (max_val - min_val of range)
    :return: float representation of how close the two values are (1 = more similar, -1 = less similar)
    """
    def compare(a: np.ndarray[any], b: np.ndarray[any]) -> float:
        if len(a) == 0 or len(b) == 0:
            return 0.0

        return float(1 - 2.0*abs(a[0]-b[0]) / (n - 1))

    return compare
