from __future__ import annotations
from nptyping import NDArray, Shape, UInt8
from dataclasses import dataclass
from numpy import (
    uint8,
    uint16,
    float32,
    double as npdouble,
    bool_,
    empty,
    zeros as npzeros,
    sum as npsum,
    dtype as npdtype
)
from numpy.random import choice
from typing import (
    Annotated,
    Final,
    Union,
    Optional,
    Any
)
from numba import (
    njit,
    types,
    prange
)

from .configuration import Camera
from ..db import CameraModel

from .extraction import (
    ChannelData,
    ColourTag,
    get_datasets_vstack,
    frequency_distribution
)


# Constants
DEFAULT_KERNEL_SIZE: Final[uint8] = uint8(5)
DEFAULT_EPSILON: Final[float32] = float32(1e-10)
DEFAULT_CHUNK_SIZE: Final[uint16] = uint16(1024)
BOUNDARY_WIDTH: Final[uint8] = uint8(5)


type BoundaryArray = Annotated[
    NDArray[Shape["*, 2"], UInt8],
    "Represents a 2D array of boundary values (N, 2)"
]
"""
Represents a 2D array of boundary values (N, 2)"]
"""

type BitMapArray =  Annotated[
    NDArray[Shape["*, *, *"], bool_],
    "Represents a 3D array of boolean values (H, W, C)"
]

JaccardRecord = npdtype([
            ('component', 'U15'), 
            ('score', float32), 
            ('index', uint8)
        ])
"""
Represents a record for Jaccard similarity scores.
Fields:
- component: str - Name of the component.
- score: float32 - Jaccard similarity score.
- index: uint8 - Index of the channel (0-2).
"""


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """
    Immutable container for ROC analysis metrics.
    """
    true_positive_rate: float32
    false_positive_rate: float32
    precision: float32
    accuracy: float32
    from dataclasses import dataclass
    def __post_init__(self):
        if not (0 <= self.true_positive_rate <= 1):
            raise ValueError("true_positive_rate must be between 0 and 1")
        if not (0 <= self.false_positive_rate <= 1):
            raise ValueError("false_positive_rate must be between 0 and 1")
        if not (0 <= self.precision <= 1):
            raise ValueError("precision must be between 0 and 1")
        if not (0 <= self.accuracy <= 1):
            raise ValueError("accuracy must be between 0 and 1")
        

@dataclass(frozen=True, slots=True)
class BoundaryRange:
    """
    """
    upper: uint8
    lower: uint8

    def __post_init__(self):
        if self.upper < self.lower:
            raise ValueError("upper boundary must be greater than or equal to lower boundary")
        if not (0 <= self.lower <= 255):
            raise ValueError("lower boundary must be between 0 and 255")
        if not (0 <= self.upper <= 255):
            raise ValueError("upper boundary must be between 0 and 255")
        if not (self.upper - self.lower) < BOUNDARY_WIDTH:
            raise ValueError("upper and lower boundaries must be within 5 units of each other")
    

@dataclass(slots=True)
class AnalysisConfiguration:
    """
    Configuration container for ROC calulcation details.
    """
    strata_count: uint16
    strata_size: uint16
    chunk_size: uint16 = DEFAULT_CHUNK_SIZE
    boundary_width: uint16 = BOUNDARY_WIDTH
    jaccard_threshold: float32 = 0.25
    max_workers: Optional[uint8] = 2
    enable_caching: bool = True

    def __post_init__(self):
        if self.strata_count < 1:
            raise ValueError("strata_count must be at least 1")
        if self.strata_size < 1:
            raise ValueError("strata_size must be at least 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if self.boundary_width < 1:
            raise ValueError("boundary_width must be at least 1")
        if not (0 <= self.jaccard_threshold <= 1):
            raise ValueError("jaccard_threshold must be between 0 and 1")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1 if specified")
    

@njit(
    types.Array(types.uint8, 2, 'C')(
        types.int32,
        types.int32
    ),
    cache=True
)
def generate_boundary_permutations(
    min_width: int = BOUNDARY_WIDTH,
    step_size: int = BOUNDARY_WIDTH
) -> BoundaryArray:
    """
    JIT-compiled boundary generation for performance.
    Internal function called by generate_boundary_permutations.
    Args:
        min_width (int): Minimum width of the boundary.
        step_size (int): Step size for generating boundaries.
    Returns:
        BoundaryArray: 2D array of boundary values.
    """
    # Pre-calculate maximum possible combinations
    max_combinations = 0
    for lower in range(0, 256, step_size):
        for upper in range(lower + min_width, 256, step_size):
            max_combinations += 1
    
    boundaries = npzeros((max_combinations, 2), dtype=uint8)
    count = 0
    
    for lower in range(0, 256, step_size):
        for upper in range(lower + min_width, 256, step_size):
            boundaries[count, 0] = lower
            boundaries[count, 1] = upper
            count += 1
    # Ensure we return only the filled part of the array
    return boundaries[:count]


@njit(
    types.float32(
        types.Array(types.uint8, 1, 'C'),
        types.Array(types.uint8, 1, 'C')
    ),
    cache=True
)
def compute_jaccard_similarity(
    array1: ChannelData, 
    array2: ChannelData
) -> float:
    """
    Compute Jaccard similarity coefficient between two arrays.

    Args:
        array1: First comparison array
        array2: Second comparison array
        
    Returns:
        Jaccard similarity coefficient [0, 1]
    """
    if len(array1) == 0 and len(array2) == 0:
        return 1.0
    
    set1 = set(array1)
    set2 = set(array2)
    
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    
    return types.float32(intersection_size / (union_size + DEFAULT_EPSILON))



@njit(
    types.containers.UniTuple(types.double, 4)(
        types.npytypes.Array(types.boolean, 3, "C"),
        types.npytypes.Array(types.boolean, 3, "C"),
    )
)
def compute_confusion_matrix(
    ground_truth_masks: BitMapArray,
    predicted_masks: BitMapArray,
) -> tuple[float32, float32, float32, float32]:
    """
    Compute confusion matrix metrics for ROC analysis.

    Args:
        ground_truth_masks: Ground truth binary masks.
        predicted_masks: Predicted binary masks.

    Returns:
        ConfusionMatrix: A tuple containing true positive rate, false positive rate,
                         precision, and accuracy.
    """
    tp = npsum(ground_truth_masks & predicted_masks)
    fn = npsum(ground_truth_masks & ~predicted_masks)
    fp = npsum(~ground_truth_masks & predicted_masks)
    tn = npsum(~ground_truth_masks & ~predicted_masks)
    
    epsilon = DEFAULT_EPSILON
    tpr = tp / (tp + fn + epsilon)
    fpr = fp / (fp + tn + epsilon)
    precision = tp / (tp + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    return float32(tpr), float32(fpr), float32(precision), float32(accuracy)
    


@njit(
    types.npytypes.Array(types.uint16, 2, "C")(
        types.npytypes.Array(types.uint16, 1, "C"), types.uint8, types.uint8
    ),
    parallel=False,  # Runs faster without parallelization. Overhead too much.
    fastmath=False,  # No difference in speed
)
def bootstrap_indexes(
    indexes: NDArray[Shape["*"], uint16],
    stratum_size: Optional[uint8] = None,
    strata_count: Optional[uint8] = 100,
) -> NDArray[uint16]:
    """
    Split the dataset indexes into testing strata using bootstrapping.
    Args:
        - indexes (List[uint16]): List of indexes to the dataset of images.
        - stratum_size (uint16, optional): Number of items in each stratum. If None, the size is set to the total number of samples in the dataset.
        - n_bootstraps (uint16, optional): Number of bootstrap iterations.
    Returns:
        NDArray[(uint8, 2)]: 2D array where each row is a bootstrap sample of indices.
    """
    n_samples = indexes.shape[0]
    if stratum_size is None:
        stratum_size = (
            n_samples  # Default to the full size of the dataset if not specified
        )
    # Initialize an empty array to store the bootstrap samples
    testing_strata = empty((strata_count, stratum_size), dtype=uint16)
    for i in range(strata_count):
        # Randomly select indices with replacement
        test_indices = choice(n_samples, size=stratum_size, replace=True)
        # Map selected indices to the original indexes and store them in the array
        testing_strata[i] = indexes[test_indices]
    return testing_strata




class ColorSpaceAnalyzer:
    """
    Class for analyzing color spaces
    """

    __slots__ = ("camera", "_cache")

    def __init__(self, camera: Camera):
        self.camera = camera
        self._cache: dict[str, Any] = {}


    def analyze_channel(self, ctag: ColourTag):
        key = ctag.tag
        if key in self._cache:
            return self._cache[key]
        
        results = npzeros(3, dtype=JaccardRecord)
        try:
            clouds, skies = get_datasets_vstack(self.camera)
            cloud_dist = frequency_distribution(clouds, ctag)
            sky_dist = frequency_distribution(skies, ctag)

            for index, component in enumerate(ctag.components):
                cloud_channel = cloud_dist[index][:, 0]
                sky_channel = sky_dist[index][:, 0]
                score = compute_jaccard_similarity(cloud_channel, sky_channel)
                results[index] = (component, score, index)
            results.sort(order='score')
            self._cache[key] = results
            return results
        
        except ValueError as err:
            raise ValueError(f"Failed to analyze '{ctag.tag} colorspace': {err}")




