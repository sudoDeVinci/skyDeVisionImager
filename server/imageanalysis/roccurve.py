from __future__ import annotations
from nptyping import NDArray, Shape, UInt8, Bool, UInt16, Object, Void
from typing_extensions import Self
from dataclasses import dataclass
from numpy import (
    uint8,
    uint16,
    float32,
    float64,
    empty,
    nan,
    zeros as npzeros,
    sum as npsum,
    dtype as npdtype,
    unique,
    array,
)
from numpy.random import choice
from numba import njit, prange, types as nbtypes
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
    get_masks_vstacks_sparse,
    get_datasets_vstacks_sparse,
    get_reference_vstacks_sparse
)

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import cv2

cv2.ocl.setUseOpenCL(True)

basicConfig(
    level="DEBUG",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler(), FileHandler("image_analysis.log", mode="w")],
)

LOGGER: Logger = getLogger("imanalysis")
numbassalog = getLogger("numba.core.ssa")
numbassalog.setLevel("WARNING")
numbabyteflow = getLogger("numba.core.byteflow")
numbabyteflow.setLevel("WARNING")
numbainterpreter = getLogger("numba.core.interpreter")
numbainterpreter.setLevel("WARNING")

# Constants
DEFAULT_KERNEL_SIZE: Final[uint8] = uint8(5)
DEFAULT_EPSILON: Final[float32] = float32(1e-32)
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
            raise ValueError(f"TPR must be between 0 and 1, got {self.true_positive_rate}")
        if not (0 <= self.false_positive_rate <= 1):
            raise ValueError(f"FPR must be between 0 and 1, got {self.false_positive_rate}")
        if not (0 <= self.precision <= 1):
            raise ValueError(f"Precision must be between 0 and 1, got {self.precision}")
        if not (0 <= self.accuracy <= 1):
            raise ValueError(f"Accuracy must be between 0 and 1, got {self.accuracy}")


@dataclass(frozen=True, slots=True)
class BoundaryRange:
    """ """

    upper: uint8
    lower: uint8

    def __post_init__(self):
        if self.upper < self.lower:
            raise ValueError(
                f"upper boundary must be greater than or equal to lower boundary :: {self.lower} & {self.upper}"
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
    boundary_width: uint8 = BOUNDARY_WIDTH
    jaccard_threshold: float32 = float32(0.25)
    max_workers: Optional[uint8] = uint8(2)
    enable_caching: bool = True

    def __post_init__(self):
        if self.strata_count < 1:
            raise ValueError("strata_count must be at least 1")
        if self.strata_size < 1:
            raise ValueError("strata_size must be at least 1")
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
            boundaries[count] = [lower, upper]
            count += 1
            
    return boundaries[:count]


@njit(
    nbtypes.float32(
        nbtypes.Array(nbtypes.uint8, 1, "C"),
        nbtypes.Array(nbtypes.uint8, 1, "C"),
    ),
    fastmath=True
)
def compute_jaccard_similarity(array1: ChannelData, array2: ChannelData) -> float32:
    """
Compute Jaccard similarity coe between two arrays using Numba-compatible operations.
    This replaces the problematic set-based implementation.

    Args:
        array1: First comparison array
        array2: Second comparison array

    Returns:
        Jaccard similarity coefficient [0, 1]
ficient between two arrays using Numba-compatible operations.
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

    unique1 = unique(array1)
    unique2 = unique(array2)

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

    union_count += (len(unique1) - i) + (len(unique2) - j)

    if union_count == 0:
        return float32(0.0)

    return float32(intersection_count / (union_count + DEFAULT_EPSILON))

@njit(
    nbtypes.UniTuple(nbtypes.float32, 4)(
        nbtypes.Array(nbtypes.bool_, 2, "C"),
        nbtypes.Array(nbtypes.bool_, 2, "C"),
    ),
    fastmath=True
)
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
    tp = (ground_truth_masks & predicted_masks).sum(dtype=float32)
    fn = (ground_truth_masks & ~predicted_masks).sum(dtype=float32)
    fp = (~ground_truth_masks & predicted_masks).sum(dtype=float32)
    tn = (~ground_truth_masks & ~predicted_masks).sum(dtype=float32)

    tpr = tp / (tp + fn + DEFAULT_EPSILON)
    fpr = fp / (fp + tn + DEFAULT_EPSILON)
    precision = tp / (tp + fp + DEFAULT_EPSILON)
    accuracy = (tp + tn) / (tp + tn + fp + fn + DEFAULT_EPSILON)

    return (tpr, fpr, precision, accuracy)


def gpu_compute_confusion_matrix(
    ground_truth_masks: cv2.UMat,
    predicted_masks: cv2.UMat,
) -> tuple[float32, float32, float32, float32]:
    
    # Create binary masks for different combinations on GPU
    # True Positive: gt=255 AND pred=255
    tp_mask_gpu = cv2.bitwise_and(ground_truth_masks, predicted_masks)
    
    # False Negative: gt=255 AND pred=0 (gt AND NOT pred)
    pred_inv_gpu = cv2.bitwise_not(predicted_masks)
    fn_mask_gpu = cv2.bitwise_and(ground_truth_masks, pred_inv_gpu)
    
    # False Positive: gt=0 AND pred=255 (NOT gt AND pred)
    gt_inv_gpu = cv2.bitwise_not(ground_truth_masks)
    fp_mask_gpu = cv2.bitwise_and(gt_inv_gpu, predicted_masks)
    
    # True Negative: gt=0 AND pred=0 (NOT gt AND NOT pred)
    tn_mask_gpu = cv2.bitwise_and(gt_inv_gpu, pred_inv_gpu)
    
    # Count non-zero pixels (GPU operations)
    tp = float32(cv2.countNonZero(tp_mask_gpu))
    fn = float32(cv2.countNonZero(fn_mask_gpu))
    fp = float32(cv2.countNonZero(fp_mask_gpu))
    tn = float32(cv2.countNonZero(tn_mask_gpu))
    
    # Calculate metrics with epsilon to avoid division by zero
    tpr = tp / (tp + fn + DEFAULT_EPSILON)
    fpr = fp / (fp + tn + DEFAULT_EPSILON)
    precision = tp / (tp + fp + DEFAULT_EPSILON)
    accuracy = (tp + tn) / (tp + tn + fp + fn + DEFAULT_EPSILON)

    return (tpr, fpr, precision, accuracy)


def bootstrap_indexes(
    indexes: NDArray[Shape["*"], UInt16],
    stratum_size: Optional[uint8] = None,
    strata_count: uint8 = uint8(100),
) -> NDArray[Shape["*, *"], UInt16]:
    """
Split the dataset indexes into testing strata using bootstrapping.
    Args:
        - indexes (List[uint16]): List of indexes to the dataset of images.
        - stratum_size (uint16, optional): Number of items in each stratum. If None, the size is set to the total number of samples in the dataset.
        - n_bootstraps (uint16, optional): Number of bootstrap iterations.
    Returns:
        NDArray[(uint8, 2)]: 2D array where each row is a bootstrap sample of indices.
    """

    n_population = len(indexes) - 1

    if stratum_size is None:
        stratum_size = uint8(n_population)
    if stratum_size > n_population:
        stratum_size = uint8(n_population)

    testing_strata = npzeros((strata_count, stratum_size), dtype=uint16)
    
    for i in range(strata_count):
        testing_strata[i] = choice(indexes, size=stratum_size, replace=True)

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

@njit
def cpu_threshold(
    image: ColorImage, channelindex: int, bounds: NDArray[Shape["2"], UInt8]
) -> BitMapImage:
    """
    Apply a threshold to the image channel based on the boundary range.
    """
    channel = image[:, :, channelindex]
    return (channel >= bounds[0]) & (channel <= bounds[1])


def gpu_threshold(image: cv2.UMat, channel_idx: int, bounds: NDArray[Shape["2"], UInt8]) -> cv2.UMat:
    """Use OpenCV's OpenCL backend"""
    channel = cv2.split(image)[channel_idx]
    lbl = cv2.UMat(array([bounds[0]], dtype=uint8))
    ub = cv2.UMat(array([bounds[1]], dtype=uint8))
    return cv2.inRange(channel, lbl, ub)

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
                boundary_width=10,
                jaccard_threshold=float32(0.15),
                max_workers=uint8(4),
                enable_caching=False,
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


    def run_boundaries(
        self: Self,
        index: uint8,
        boundaries: NDArray[Shape["*, 2"], UInt8],
        groundtruth_masks: cv2.UMat,
        reference_image: cv2.UMat
    ) -> NDArray[Shape["*, 6"], float32]:
        
        results = npzeros((boundaries.shape[0], 6), dtype=float32)
        for i in range(boundaries.shape[0]):
            thresholded_masks = gpu_threshold(reference_image, int(index), boundaries[i])
            tpr, fpr, precision, accuracy = gpu_compute_confusion_matrix(groundtruth_masks, thresholded_masks)
            lower, upper = boundaries[i]
            results[i] = [float32(lower), float32(upper), tpr, fpr, precision, accuracy]
            

        return results


    def _analyze_channel_roc(
        self: Self,
        camera: Camera,
        tag: ColourTag,
        index: uint8,
        bootstrap_samples: NDArray[Shape["*, *"], UInt16],
        boundaries: BoundaryArray
    ) -> NDArray[Shape["*, 6"], float64]:
        
        n_strata = bootstrap_samples.shape[0]

        LOGGER.debug(f"Analyzing channel {tag.tag} index {index} for camera {camera.model.value}")

        # Build the massive concatenated image on GPU
        gt_mask_umats = []
        ref_img_umats = []
        
        for stratum_index in range(n_strata):
            sample_indices = bootstrap_samples[stratum_index]

            # Get CPU data first
            gt_masks, _ = get_masks_vstacks_sparse(camera, sample_indices)
            ref_imgs = get_reference_vstacks_sparse(camera, tag, sample_indices)

            if gt_masks is None or ref_imgs is None:
                LOGGER.error(f"Failed to load data for stratum {stratum_index}")
                continue
            
            # Convert and move to GPU immediately
            gt_masks_uint8 = (gt_masks * 255).astype(uint8)
            gt_umat = cv2.UMat(gt_masks_uint8)
            ref_umat = cv2.UMat(ref_imgs)
            
            gt_mask_umats.append(gt_umat)
            ref_img_umats.append(ref_umat)

        # Horizontal concatenation on GPU
        if len(gt_mask_umats) > 1:
            big_gt_umat = cv2.hconcat(gt_mask_umats)
            big_ref_umat = cv2.hconcat(ref_img_umats)
        else:
            big_gt_umat = gt_mask_umats[0]
            big_ref_umat = ref_img_umats[0]
        
        #LOGGER.debug(f"Created massive GPU image: {big_ref_umat.shape}, GT masks: {big_gt_umat.shape}")

        # Convert UMats back to numpy for run_boundaries compatibility
        #big_gt_numpy = (big_gt_umat.get() > 0).astype(bool)  # Convert uint8 back to bool
        #big_ref_numpy = big_ref_umat.get()
        
        results = self.run_boundaries(
            index=index,
            boundaries=boundaries, 
            groundtruth_masks=big_gt_umat,
            reference_image=big_ref_umat
        )

        LOGGER.debug(f"Completed channel analysis for {tag.tag} index {index}")
        return results.astype(float64)  # Convert to expected return type

    def analyze_roc(
        self: Self,
        camera: Camera,
        colortags: list[ColourTag],
        overwrite: bool = False
    ) -> dict[str, NDArray[Shape["6"], float64]]:
        
        LOGGER.info(f"Starting ROC analysis for camera {camera.model.value}")

        similarityresults = self.run_similarity_analysis(camera, colortags)
        if not similarityresults.size:
            LOGGER.warning("No similarity results found")
            return {}
        
        bootstrap_samples = bootstrap_indexes(
            array([i for i in range(len(camera.cloud_images_paths()))]),
            stratum_size=self.config.strata_size,
            strata_count=self.config.strata_count
        )

        boundaries = generate_boundary_permutations(
            min_width=self.config.boundary_width,
            step_size=self.config.boundary_width
        )

        results = {}

        # Process each color tag that passed similarity analysis
        for similarity_result in similarityresults:
            ctag_name = similarity_result['tag']
            
            # Find the best channel (lowest Jaccard score = most discriminative)
            best_channel = similarity_result['components'][0]  # Already sorted by score
            
            if best_channel['score'] > self.config.jaccard_threshold:
                LOGGER.info(f"Skipping {ctag_name} - similarity too high: {best_channel['score']}")
                continue
                
            channel_index = best_channel['index']
            tag = ColourTag.match(ctag_name)
            if tag is ColourTag.UNKNOWN:
                LOGGER.warning(f"Unknown color tag: {ctag_name}, skipping analysis")
                continue
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future = executor.submit(
                    self._analyze_channel_roc,
                    camera,
                    tag,
                    channel_index,
                    bootstrap_samples,
                    boundaries
                )
                
        try:
            channel_results = future.result(timeout=300)  # 5 minute timeout
            results[f"{ctag_name}_{best_channel['component']}"] = channel_results
            LOGGER.info(f"Completed analysis for {ctag_name} channel {best_channel['component']}")
            
        except Exception as e:
            LOGGER.error(f"Failed to analyze {ctag_name}: {e}")

        LOGGER.info(f"ROC analysis completed for camera {camera.model.value}")
        return results