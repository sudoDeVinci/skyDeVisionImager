from __future__ import annotations
from nptyping import NDArray, Shape, UInt8, Bool, UInt16, Object, Void
from dataclasses import dataclass
from numpy import (
    uint8,
    uint16,
    float32,
    empty,
    zeros as npzeros,
    sum as npsum,
    dtype as npdtype,
    unique,
)
from numpy.random import choice
from typing import Annotated, Final, Optional, Any, cast

from .configuration import Camera
from ..db import CameraModel
from logging import Logger, getLogger, basicConfig, StreamHandler, FileHandler

from .extraction import (
    ChannelData,
    ColourTag,
    get_datasets_vstack,
    frequency_distribution,
    BitMapImage,
    ColorImage,
)

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

basicConfig(
    level="DEBUG",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler(), FileHandler("image_analysis.log", mode="w")],
)

LOGGER: Logger = getLogger("imanalysis")


# Constants
DEFAULT_KERNEL_SIZE: Final[uint8] = uint8(5)
DEFAULT_EPSILON: Final[float32] = float32(1e-10)
DEFAULT_CHUNK_SIZE: Final[uint16] = uint16(1024)
BOUNDARY_WIDTH: Final[uint8] = uint8(5)


type BoundaryArray = Annotated[
    NDArray[Shape["*, 2"], UInt8], "Represents a 2D array of boundary values (N, 2)"
]
"""
Represents a 2D array of boundary values (N, 2)"]
"""

type BitMapArray = Annotated[
    NDArray[Shape["*, *, *"], Bool],
    "Represents a 3D array of boolean values (H, W, C)",
]

type ColorImageArray = Annotated[
    NDArray[Shape["*, *, 3"], UInt8],
    "Represents a 3D array of color images (H, W, 3) with uint8 values",
]

JaccardRecord = npdtype([("component", "U15"), ("score", float32), ("index", uint8)])
"""
Represents a record for Jaccard similarity scores.
Fields:
- component: str - Name of the component.
- score: float32 - Jaccard similarity score.
- index: uint8 - Index of the channel (0-2).
"""

ColorSpaceChannelsJaccardRecords = npdtype(
    [("tag", "U20"), ("components", JaccardRecord, 3)]
)
"""
Represents a record for Jaccard similarity scores across the channels of a color space
between the cloud and sky datasets.
Fields:
- tag: str - Color space tag.
- components: JaccardRecord - Array of Jaccard similarity scores for each channel.
Each record contains:
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
    """ """

    upper: uint8
    lower: uint8

    def __post_init__(self):
        if self.upper < self.lower:
            raise ValueError(
                "upper boundary must be greater than or equal to lower boundary"
            )
        if not (0 <= self.lower <= 255):
            raise ValueError("lower boundary must be between 0 and 255")
        if not (0 <= self.upper <= 255):
            raise ValueError("upper boundary must be between 0 and 255")
        if (self.upper - self.lower) < BOUNDARY_WIDTH:
            raise ValueError(
                f"upper and lower boundaries must be within {BOUNDARY_WIDTH} units of each other"
            )


@dataclass(slots=True)
class AnalysisConfiguration:
    """
    Configuration container for ROC calulcation details.
    """

    strata_count: uint16
    strata_size: uint16
    chunk_size: uint16 = DEFAULT_CHUNK_SIZE
    boundary_width: uint8 = BOUNDARY_WIDTH
    jaccard_threshold: float32 = float32(0.25)
    max_workers: Optional[uint8] = uint8(2)
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


def generate_boundary_permutations(
    min_width: uint8 = BOUNDARY_WIDTH, step_size: uint8 = BOUNDARY_WIDTH
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


def compute_jaccard_similarity(array1: ChannelData, array2: ChannelData) -> float32:
    """
    Compute Jaccard similarity coefficient between two arrays using Numba-compatible operations.
    This replaces the problematic set-based implementation.

    Args:
        array1: First comparison array
        array2: Second comparison array

    Returns:
        Jaccard similarity coefficient [0, 1]
    """
    if len(array1) == 0 and len(array2) == 0:
        return float32(1.0)

    if len(array1) == 0 or len(array2) == 0:
        return float32(0.0)

    # Get sorted unique values (unique() returns sorted array)
    unique1 = unique(array1)
    unique2 = unique(array2)

    # Two-pointer technique for intersection
    i, j = 0, 0
    intersection_count = 0
    union_count = 0

    while i < len(unique1) and j < len(unique2):
        if unique1[i] == unique2[j]:
            intersection_count += 1
            union_count += 1
            i += 1
            j += 1
        elif unique1[i] < unique2[j]:
            union_count += 1
            i += 1
        else:
            union_count += 1
            j += 1

    # Add remaining elements to union
    union_count += (len(unique1) - i) + (len(unique2) - j)

    if union_count == 0:
        return float32(0.0)

    return float32(intersection_count / (union_count + DEFAULT_EPSILON))


def compute_confusion_matrix(
    ground_truth_masks: BitMapImage,
    predicted_masks: BitMapImage,
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

    tpr = tp / (tp + fn + DEFAULT_EPSILON)
    fpr = fp / (fp + tn + DEFAULT_EPSILON)
    precision = tp / (tp + fp + DEFAULT_EPSILON)
    accuracy = (tp + tn) / (tp + tn + fp + fn + DEFAULT_EPSILON)

    return (tpr, fpr, precision, accuracy)


def bootstrap_indexes(
    indexes: NDArray[Shape["*"], UInt16],
    stratum_size: Optional[uint8] = None,
    strata_count: uint8 = uint8(100),
) -> NDArray[Shape["*, 2"], UInt16]:
    """
    Split the dataset indexes into testing strata using bootstrapping.
    Args:
        - indexes (List[uint16]): List of indexes to the dataset of images.
        - stratum_size (uint16, optional): Number of items in each stratum. If None, the size is set to the total number of samples in the dataset.
        - n_bootstraps (uint16, optional): Number of bootstrap iterations.
    Returns:
        NDArray[(uint8, 2)]: 2D array where each row is a bootstrap sample of indices.
    """

    n_population = indexes.shape[0]

    if stratum_size is None:
        stratum_size = uint8(n_population / 2)
    if stratum_size > n_population:
        stratum_size = uint8(n_population / 2)
    # Initialize an empty array to store the bootstrap samples
    testing_strata = npzeros(
        (
            strata_count,
            stratum_size,
        ),
        dtype=uint16,
    )
    for i in range(strata_count):  # type: ignore
        test_indices = choice(n_population, size=stratum_size, replace=True)
        testing_strata[i] = indexes[test_indices]

    return testing_strata


def analyze_channel(
    ctag: ColourTag,
    skyset: Optional[ColorImage] = None,
    cloudset: Optional[ColorImage] = None,
) -> NDArray[Shape["*, 3"], Void]:
    LOGGER.debug(f">> Analyzing color channel: {ctag.tag}")

    results = npzeros((3,), dtype=JaccardRecord)

    try:
        if skyset is None or cloudset is None:
            raise ValueError("Sky and cloud datasets must be provided for analysis.")

        cloud_dist = frequency_distribution(cloudset, ctag)
        sky_dist = frequency_distribution(skyset, ctag)

        # print(f">> Cloud distribution: {cloud_dist}, Sky distribution: {sky_dist}")

        LOGGER.debug(
            f">> Cloud distribution: {cloud_dist.shape}, Sky distribution: {sky_dist.shape}"
        )

        for index, component in enumerate(ctag.components):
            cloud_channel = cloud_dist[index][:, 0]
            sky_channel = sky_dist[index][:, 0]
            score = compute_jaccard_similarity(cloud_channel, sky_channel)
            results[index] = (component, score, index)
        LOGGER.debug(f">> Jaccard scores: {results}")
        results.sort(order="score")
        LOGGER.debug(f">> Sorted Jaccard scores: {results}")
        return results
    except ValueError as err:
        raise ValueError(f"Failed to analyze '{ctag.tag} colorspace': {err}")


class ROCAnalyzer:

    __slots__ = ("config", "camera", "_cache")

    def __init__(
        self,
        config: Optional[AnalysisConfiguration] = None,
        camera: Optional[Camera] = None,
    ):
        self.config = (
            config
            if config is not None
            else AnalysisConfiguration(
                strata_count=uint16(30),
                strata_size=uint16(30),
                chunk_size=DEFAULT_CHUNK_SIZE,
                boundary_width=BOUNDARY_WIDTH,
                jaccard_threshold=float32(0.3),
                max_workers=uint8(2),
                enable_caching=True,
            )
        )
        self._cache: dict[str, Any] = {}
        self.camera = camera

    def generate_cache_key(self, ctag: ColourTag) -> str:
        """
        Generate a cache key based on the camera model and color tag.
        Raises:
            ValueError: If camera or configuration is not set.
        Returns:
            str: A unique cache key for the analysis.
        """
        if self.camera is None or self.config is None:
            raise ValueError(
                "Camera and configuration must be set before generating cache key"
            )
        return f"{self.camera.model.value}_{ctag.tag}_{self.config.strata_count}_{self.config.strata_size}_{self.config.chunk_size}"

    def threshold(
        self, image: ColorImage, channelindex: int, boundary: BoundaryRange
    ) -> BitMapImage:
        """
        Apply a threshold to the image channel based on the boundary range.
        """
        channel = image[:, :, channelindex]
        return cast(
            BitMapImage, (channel >= boundary.lower) & (channel <= boundary.upper)
        )

    def run_similarity_analysis(
        self, camera: Camera, ctags: list[ColourTag]
    ) -> NDArray[Shape["*, 3"], Void]:
        """
        Run similarity analysis for the given color tags.
        Args:
            camera (Camera): Camera instance to use for analysis.
            ctags (list[ColourTag]): List of color tags to analyze.
        Returns:
            NDArray[Shape["*, 3"], JaccardRecord]: Array of Jaccard similarity records.
        """

        clist = [tag for tag in ctags if tag != ColourTag.UNKNOWN]

        if not clist:
            LOGGER.warning("No valid color tags to analyze.")
            return empty((1,), dtype=ColorSpaceChannelsJaccardRecords)

        results = npzeros(len(clist), dtype=ColorSpaceChannelsJaccardRecords)
        workers = self.config.max_workers or min(len(clist), 4)

        LOGGER.debug(
            f">> Running similarity analysis with {workers} workers for {len(clist)} color tags."
        )

        # Get the datasets for cloud and sky images so we can reuse across Processes
        cloudset, skyset = get_datasets_vstack(camera)

        with ProcessPoolExecutor(max_workers=int(workers)) as executor:

            # Submit tasks for each color tag - Futures are hashable so we can map them to their color tag and retrieve as_completed
            LOGGER.debug(
                f"Starting analysis for {len(clist)} color tags with {workers} workers."
            )
            LOGGER.debug(
                f"Cloud dataset size: {cloudset.shape}, Sky dataset size: {skyset.shape}"
            )

            futures = {
                executor.submit(analyze_channel, ctag, skyset, cloudset): ctag
                for ctag in clist
            }

            for future in as_completed(futures):
                ctag = futures[future]
                try:
                    result = future.result()
                    results[clist.index(ctag)] = (ctag.tag, result)
                except Exception as e:
                    LOGGER.error(f"Error analyzing {ctag.tag}: {e}")

        return results
