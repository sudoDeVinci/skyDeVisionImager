from __future__ import annotations
from nptyping import (
    NDArray,
    Shape,
    UInt8,
    UInt16,
    Void,
    Bool
)
from numpy import (
    uint8,
    uint16,
    float32,
    float64,
    empty,
    unique,
    array,
    zeros as npzeros,
    dtype as npdtype,
)

from numpy.random import choice
from numba import njit, types as nbtypes
from typing_extensions import Self
from dataclasses import dataclass
from typing import Annotated, Final, Optional, Any
import cv2

from .configuration import Camera
from logging import Logger, getLogger, basicConfig, StreamHandler, FileHandler

from .extraction import (
    ColourTag,
    ColorImage,
    BitMapImage,
    ChannelData,
    get_datasets_vstack,
    frequency_distribution,
    get_masks_vstacks_sparse,
    get_reference_vstacks_sparse
)

from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
)

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


DEFAULT_EPSILON: Final[float32] = float32(1e-32)
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


def verify_gpu_setup() -> bool:
    """
    Verify GPU setup and log OpenCL information.
    Returns True if GPU is properly configured.

    Returns:
        bool: True if GPU is properly configured, False otherwise.
    """
    LOGGER.info("=== GPU Setup Verification ===")
    
    opencl_available = cv2.ocl.haveOpenCL()
    opencl_enabled = cv2.ocl.useOpenCL()
    
    LOGGER.info(f"OpenCL available: {opencl_available}")
    LOGGER.info(f"OpenCL enabled: {opencl_enabled}")
    
    if not opencl_available:
        LOGGER.warning("OpenCL not available - falling back to CPU")
        return False
        
    if not opencl_enabled:
        LOGGER.warning("OpenCL not enabled - enabling now")
        cv2.ocl.setUseOpenCL(True)
    
    try:
        device = cv2.ocl.Device.getDefault()

        log = ("\n"
            f"\nDevice Name: {device.name()}\n"
            f"Device Type: {device.type()}\n"
            f"Max Compute Units: {device.maxComputeUnits()}\n"
            f"Global Memory: {device.globalMemSize() // (1024*1024)} MB\n"
            f"Local Memory: {device.localMemSize() // 1024} KB\n"
            f"Max Work Group Size: {device.maxWorkGroupSize()}\n"
        )

        LOGGER.info(log)

    except Exception as e:
        LOGGER.warning(f"Could not get device info: {e}")
    
    try:
        test_umat = cv2.UMat(rows=1000, cols=1000, type=cv2.CV_8UC1)
        result = cv2.blur(test_umat, (5, 5))
        
        if isinstance(result, cv2.UMat):
            LOGGER.info("GPU operations working correctly")
            return True
        else:
            LOGGER.warning("GPU operations not working - result is not UMat")
            return False
            
    except Exception as e:
        LOGGER.error(f"GPU test failed: {e}")
        return False


@dataclass(slots=True)
class AnalysisConfiguration:
    """
    Configuration for ROC calculation details.
    """

    strata_count: uint16
    strata_size: uint16
    boundary_width: uint8 = BOUNDARY_WIDTH
    jaccard_threshold: float32 = float32(0.25)
    max_workers: Optional[uint8] = uint8(2)
    caching: bool = True
    gpu_caching: bool = True

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
    step_size: uint8 = BOUNDARY_WIDTH
) -> BoundaryArray:
    """
    Generate all possible boundary permutations for thresholding.
    This generates pairs of lower and upper bounds for thresholding operations.
    Args:
        step_size (int): Step size for generating boundaries.
    Returns:
        BoundaryArray: 2D array of boundary values.
    """
    max_combinations = 0
    for lower in range(0, 256, step_size):
        for upper in range(lower + step_size, 256, step_size):
            max_combinations += 1

    boundaries = npzeros((max_combinations, 2), dtype=uint8)
    count = 0

    for lower in range(0, 256, step_size):
        for upper in range(lower + step_size, 256, step_size):
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
    Compute Jaccard similarity coeff between two arrays using Numba-compatible operations.

    Args:
        array1 (NDArray[N, uint8]): First comparison array
        array2 (NDArray[N, uint8]): Second comparison array

    Returns:
        float32: Jaccard similarity coefficient between two arrays using Numba-compatible operations.
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
def cpu_compute_confusion_matrix(
    ground_truth_masks: BitMapImage,
    predicted_masks: BitMapImage,
) -> NDArray[Shape["4"], float32]:
    """
    Compute confusion matrix metrics for ROC analysis on our cpu as a fallback.
    This one is specifically implemented for numba use.

    Args:
        ground_truth_masks: Composite image of the ground truth BitMaps corresponding to this strata.
        predicted_masks: Predicted binary masks.

    Returns:
        ConfusionMatrix: An array containing true positive rate, false positive rate, precision, and accuracy.
    """
    tp = (ground_truth_masks & predicted_masks).sum(dtype=float32)
    fn = (ground_truth_masks & ~predicted_masks).sum(dtype=float32)
    fp = (~ground_truth_masks & predicted_masks).sum(dtype=float32)
    tn = (~ground_truth_masks & ~predicted_masks).sum(dtype=float32)

    tpr = tp / (tp + fn + DEFAULT_EPSILON)
    fpr = fp / (fp + tn + DEFAULT_EPSILON)
    precision = tp / (tp + fp + DEFAULT_EPSILON)
    accuracy = (tp + tn) / (tp + tn + fp + fn + DEFAULT_EPSILON)

    return tpr, fpr, precision, accuracy


def gpu_compute_confusion_matrix(
    ground_truth_masks: cv2.UMat,
    predicted_masks: cv2.UMat,
) -> tuple[float32, float32, float32, float32]:
    """
    Compute confusion matrix metrics for ROC analysis on our GPU.

    Args:
        ground_truth_masks: Composite image of the ground truth BitMaps corresponding to this strata.
        predicted_masks: Predicted binary masks.

    Returns:
        ConfusionMatrix: An array containing true positive rate, false positive rate, precision, and accuracy.
    """
    tp_mask_gpu = cv2.bitwise_and(ground_truth_masks, predicted_masks)
    pred_inv_gpu = cv2.bitwise_not(predicted_masks)
    fn_mask_gpu = cv2.bitwise_and(ground_truth_masks, pred_inv_gpu)
    gt_inv_gpu = cv2.bitwise_not(ground_truth_masks)
    fp_mask_gpu = cv2.bitwise_and(gt_inv_gpu, predicted_masks)
    tn_mask_gpu = cv2.bitwise_and(gt_inv_gpu, pred_inv_gpu)
    
    tp = float32(cv2.countNonZero(tp_mask_gpu))
    fn = float32(cv2.countNonZero(fn_mask_gpu))
    fp = float32(cv2.countNonZero(fp_mask_gpu))
    tn = float32(cv2.countNonZero(tn_mask_gpu))
    
    tpr = tp / (tp + fn + DEFAULT_EPSILON)
    fpr = fp / (fp + tn + DEFAULT_EPSILON)
    precision = tp / (tp + fp + DEFAULT_EPSILON)
    accuracy = (tp + tn) / (tp + tn + fp + fn + DEFAULT_EPSILON)

    return tpr, fpr, precision, accuracy


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


def analyze_channel_jaccard(
    ctag: ColourTag,
    skyset: Optional[ColorImage] = None,
    cloudset: Optional[ColorImage] = None,
) -> NDArray[Shape["*, 3"], Void]:
    """
    Analyze the Jaccard similarity between the cloud and sky datasets for a specific color channel.

    Args:
        ctag (ColourTag): The color tag to analyze.
        skyset (Optional[ColorImage]): The sky dataset to compare against.
        cloudset (Optional[ColorImage]): The cloud dataset to analyze.
    Returns:
        NDArray[Shape["*, 3"], JaccardRecord]: Array of Jaccard similarity records for the specified color channel.
    Raises:
        ValueError: If skyset or cloudset is None.
    """

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
    """
    
    """

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
                boundary_width=5,
                jaccard_threshold=float32(0.25),
                max_workers=uint8(6),
                caching=False,
                gpu_caching=True
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
        return f"{self.camera.model.value}_{ctag.tag}_{self.config.strata_count}_{self.config.strata_size}"


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
                executor.submit(analyze_channel_jaccard, ctag, skyset, cloudset): ctag
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
        
        #results = npzeros((boundaries.shape[0], 6), dtype=float32)
        
        # Pre-split the reference image once to avoid repeated splits
        target_channel = cv2.split(reference_image)[int(index)]
        results = npzeros((boundaries.shape[0], 6), dtype=float32)

        for i in range(1, boundaries.shape[0]):
            LOGGER.debug(f"Processing boundary {i+1}/{boundaries.shape[0]}: {boundaries[i]}")
            lower_bound = int(boundaries[i][0])
            upper_bound = int(boundaries[i][1])

            # Use the pre-split channel for more efficient thresholding
            thresholded_masks = cv2.inRange(target_channel, lower_bound, upper_bound)
            tpr, fpr, precision, accuracy = gpu_compute_confusion_matrix(groundtruth_masks, thresholded_masks)
            results[i] = [lower_bound, upper_bound, tpr, fpr, precision, accuracy]

        return results


    def _load_batch_to_gpu(
        self,
        camera: Camera,
        tag: ColourTag,
        bootstrap_samples: NDArray[Shape["*, *"], UInt16]
    ) -> tuple[cv2.UMat, cv2.UMat]:
        """
        Load all bootstrap data directly to GPU in batches to minimize transfers.
        """
        
        LOGGER.debug(f"Loading {bootstrap_samples.shape[0]} strata directly to GPU")
        
        # Load first sample to get dimensions
        sample_indices = bootstrap_samples[0]
        gt_sample, _ = get_masks_vstacks_sparse(camera, sample_indices[:1])
        ref_sample = get_reference_vstacks_sparse(camera, tag, sample_indices[:1])
        
        h, w = gt_sample.shape
        _, _, c = ref_sample.shape
        
        # Pre-allocate large GPU buffers
        total_height = h * bootstrap_samples.shape[0] * bootstrap_samples.shape[1]
        
        # Create large CPU buffers first
        all_gt_cpu = npzeros((total_height, w), dtype=uint8)
        all_ref_cpu = npzeros((total_height, w, c), dtype=uint8)
        
        current_row = 0
        
        # Fill CPU buffers efficiently
        for stratum_idx in range(bootstrap_samples.shape[0]):
            sample_indices = bootstrap_samples[stratum_idx]
            
            gt_masks, _ = get_masks_vstacks_sparse(camera, sample_indices)
            ref_imgs = get_reference_vstacks_sparse(camera, tag, sample_indices)
            
            if gt_masks is not None and ref_imgs is not None:
                gt_uint8 = (gt_masks * 255).astype(uint8)
                
                rows_to_add = gt_uint8.shape[0]
                all_gt_cpu[current_row:current_row + rows_to_add] = gt_uint8
                all_ref_cpu[current_row:current_row + rows_to_add] = ref_imgs
                current_row += rows_to_add
        
        # Single transfer to GPU
        final_gt_umat = cv2.UMat(all_gt_cpu[:current_row])
        final_ref_umat = cv2.UMat(all_ref_cpu[:current_row])
        
        
        LOGGER.debug(f"Transferred {current_row} rows to GPU - GT: ({current_row}, {w}), Ref: ({current_row}, {w}, {c})")
        return final_gt_umat, final_ref_umat

    def _analyze_channel_roc(
        self: Self,
        camera: Camera,
        tag: ColourTag,
        index: uint8,
        bootstrap_samples: NDArray[Shape["*, *"], UInt16],
        boundaries: BoundaryArray | cv2.UMat
    ) -> NDArray[Shape["*, 6"], float64]:
        
        LOGGER.debug(f"Analyzing channel {tag.tag} index {index} for camera {camera.model.value}")

        big_gt_umat, big_ref_umat = self._load_batch_to_gpu(camera, tag, bootstrap_samples)
        
        LOGGER.debug(f"GPU batch loaded successfully")

        results = self.run_boundaries(
            index=index,
            boundaries=boundaries, 
            groundtruth_masks=big_gt_umat,
            reference_image=big_ref_umat
        )

        LOGGER.debug(f"Completed channel analysis for {tag.tag} index {index}")
        return results

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
            step_size=self.config.boundary_width
        )

        results = {}

        try:
            analysis_tasks = []
            for similarity_result in similarityresults:
                ctag_name = similarity_result['tag']
                
                best_channel = similarity_result['components'][0]
                
                if best_channel['score'] > self.config.jaccard_threshold:
                    LOGGER.info(f"Skipping {ctag_name} - similarity too high: {best_channel['score']}")
                    continue
                    
                channel_index = best_channel['index']
                tag = ColourTag.match(ctag_name)
                if tag is ColourTag.UNKNOWN:
                    LOGGER.warning(f"Unknown color tag: {ctag_name}, skipping analysis")
                    continue
                
                analysis_tasks.append({
                    'ctag_name': ctag_name,
                    'tag': tag,
                    'channel_index': channel_index,
                    'best_channel': best_channel
                })
            
            if not analysis_tasks:
                LOGGER.warning("No valid analysis tasks found")
                return {}
            
            LOGGER.info(f"Starting ROC analysis for {len(analysis_tasks)} color channels")
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                for task in analysis_tasks:
                    future = executor.submit(
                        self._analyze_channel_roc,
                        camera,
                        task['tag'],
                        task['channel_index'],
                        bootstrap_samples,
                        boundaries
                    )
                    futures[future] = task
                
                for future in as_completed(futures, timeout=600):
                    task = futures[future]
                    ctag_name = task['ctag_name']
                    best_channel = task['best_channel']
                    
                    try:
                        channel_results = future.result()
                        results[f"{ctag_name}_{best_channel['component']}"] = channel_results
                        LOGGER.info(f"Completed analysis for {ctag_name} channel {best_channel['component']}")
                        
                    except Exception as e:
                        LOGGER.error(f"Failed to analyze {ctag_name}: {e}")

            LOGGER.info(f"ROC analysis completed for camera {camera.model.value}")
            return results
    
        except Exception as e:
            LOGGER.error(f"Error during ROC analysis: {e}")
            return {}
    