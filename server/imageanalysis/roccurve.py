from __future__ import annotations
from nptyping import NDArray, Shape, UInt8, UInt16, Float32, Float64, Void, Bool
from numpy import (
    uint8,
    uint16,
    float32,
    float64,
    empty,
    unique,
    array,
    ascontiguousarray,
    zeros as npzeros,
    dtype as npdtype,
    copy as npcopy,
)

from numpy.random import choice
from numba import njit, types as nbtypes  # type: ignore[import-untyped]
from typing_extensions import Self
from dataclasses import dataclass
from typing import Annotated, Final, Optional, Any
import cv2

from .configuration import Camera, LOGGER
from logging import getLogger
from .extraction import (
    ColourTag,
    ColorImage,
    BitMapImage,
    GrayScaleImage,
    ChannelData,
    get_datasets_vstack,
    frequency_distribution,
    get_masks_vstacks_sparse,
    get_reference_vstacks_sparse,
)

from concurrent.futures import (
    as_completed,
    ThreadPoolExecutor,  # Changed from ProcessPoolExecutor
)

numbassalog = getLogger("numba.core.ssa")
numbassalog.setLevel("WARNING")
numbabyteflow = getLogger("numba.core.byteflow")
numbabyteflow.setLevel("WARNING")
numbainterpreter = getLogger("numba.core.interpreter")
numbainterpreter.setLevel("WARNING")


DEFAULT_EPSILON: Final[float32] = float32(1e-32)
BOUNDARY_WIDTH: Final[uint8] = uint8(10)


type BoundaryArray = list[tuple[uint8, list[uint8]]]
"""
A list of tuple boundaries for thresholding. The first item is the lower boundary, the second item is a list of upper boundaries.
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


@dataclass(slots=True)
class AnalysisConfiguration:
    """
    Configuration for ROC calculation details.
    """

    strata_count: uint16
    strata_size: uint16
    boundary_width: uint8 = BOUNDARY_WIDTH
    jaccard_threshold: float32 = float32(0.25)
    max_workers: uint8 = uint8(2)
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

    def to_dict(self):
        """
        Convert the dataclass to a dictionary for easy serialization.
        """
        return {
            "strata_count": int(self.strata_count),
            "strata_size": int(self.strata_size),
            "boundary_width": int(self.boundary_width),
            "jaccard_threshold": float(self.jaccard_threshold),
            "max_workers": int(self.max_workers),
            "caching": self.caching,
            "gpu_caching": self.gpu_caching,
        }


def generate_boundary_permutations(step_size: uint8 = BOUNDARY_WIDTH) -> BoundaryArray:
    """
    Generate all possible boundary permutations for thresholding.
    This generates f lower and upper bounds for thresholding operations.
    Args:
        step_size (int): Step size for generating boundaries.
    Returns:
        A list of tuple boundaries for thresholding. The first item is the lower boundary, the second item is a list of upper boundaries.
    """

    boundaries: BoundaryArray = []

    for lower in range(0, 256, step_size):
        boundarytuple: tuple[uint8, list[uint8]] = (uint8(lower), [])
        for upper in range(lower + step_size, 256, step_size):
            boundarytuple[1].append(uint8(upper))
        if boundarytuple[1]:
            boundaries.append(boundarytuple)

    return boundaries


@njit(
    nbtypes.float32(
        nbtypes.Array(nbtypes.uint8, 1, "C"),
        nbtypes.Array(nbtypes.uint8, 1, "C"),
    ),
    fastmath=True,
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


def _compute_confusion_matrix_masks(
    ground_truth_masks: cv2.UMat,
    predicted_masks: cv2.UMat,
) -> tuple[cv2.UMat, cv2.UMat, cv2.UMat, cv2.UMat]:
    """
    Compute confusion matrix metrics for ROC analysis on our GPU.
    Args:
        ground_truth_masks (cv2.UMat): Ground truth masks as UMat.
        predicted_masks (cv2.UMat): Predicted masks as UMat.
    Returns:
        tuple: A tuple containing the true positive, false negative, false positive, and true negative masks.
    """
    tp_mask = cv2.bitwise_and(ground_truth_masks, predicted_masks)
    pred_inv = cv2.bitwise_not(predicted_masks)
    fn_mask = cv2.bitwise_and(ground_truth_masks, pred_inv)
    gt_inv = cv2.bitwise_not(ground_truth_masks)
    fp_mask = cv2.bitwise_and(gt_inv, predicted_masks)
    tn_mask = cv2.bitwise_and(gt_inv, pred_inv)

    return tp_mask, fn_mask, fp_mask, tn_mask


def _compute_confusion_matrix(
    tp_mask: cv2.UMat,
    fn_mask: cv2.UMat,
    fp_mask: cv2.UMat,
    tn_mask: cv2.UMat,
):
    tp = float32(cv2.countNonZero(tp_mask))
    fn = float32(cv2.countNonZero(fn_mask))
    fp = float32(cv2.countNonZero(fp_mask))
    tn = float32(cv2.countNonZero(tn_mask))

    tpr = tp / (tp + fn + DEFAULT_EPSILON)
    fpr = fp / (fp + tn + DEFAULT_EPSILON)
    precision = tp / (tp + fp + DEFAULT_EPSILON)
    accuracy = (tp + tn) / (tp + tn + fp + fn + DEFAULT_EPSILON)

    return tpr, fpr, precision, accuracy


def _batch_compute_confusion_matrix(
    lower_bound: uint8,
    upper_bounds: list[uint8],
    mask_shape: tuple[int, int],
    ground_truth_masks: cv2.UMat,
    boundary_masks: cv2.UMat,
) -> NDArray[Shape["*, 5"], Float32]:

    boundary_count = len(upper_bounds)
    mask_height, mask_width = mask_shape
    # Replicate the ground truth masks for each boundary
    results = empty((boundary_count, 6), dtype=float32)
    repeated_ground_truth_masks = cv2.repeat(ground_truth_masks, boundary_count, 1)

    LOGGER.debug(
        f"_batch_compute_confusion_matrix :: Lower bound: {lower_bound}, Upper bounds: {upper_bounds}, Mask shape: {mask_shape}"
    )
    LOGGER.debug(
        f"_batch_compute_confusion_matrix :: Boundary count: {boundary_count}, Mask height: {mask_height}, Mask width: {mask_width}"
    )

    # Precompute all the confusion matrix components at once
    LOGGER.debug(
        f"_batch_compute_confusion_matrix :: Computing confusion matrix masks for {boundary_count} boundaries"
    )
    tpmask, fnmask, fpmask, tnmask = _compute_confusion_matrix_masks(
        repeated_ground_truth_masks, boundary_masks
    )
    LOGGER.debug(f"_batch_compute_confusion_matrix :: Confusion matrix masks computed")

    for i in range(boundary_count):
        startrow = i * mask_height

        roi: cv2.typing.Rect = (0, startrow, mask_width, mask_height)

        # Make a mask for this boundary stripe
        tpslice = cv2.UMat(tpmask, roi)
        fnslice = cv2.UMat(fnmask, roi)
        fpslice = cv2.UMat(fpmask, roi)
        tnslice = cv2.UMat(tnmask, roi)

        LOGGER.debug(
            f"_batch_compute_confusion_matrix :: Processing boundary {i + 1}/{boundary_count} with bounds ({lower_bound}, {upper_bounds[i]})"
        )
        tpr, fpr, precision, accuracy = _compute_confusion_matrix(
            tpslice,
            fnslice,
            fpslice,
            tnslice,
        )

        LOGGER.debug(
            f"_batch_compute_confusion_matrix :: Boundary {i + 1} results: TPR={tpr}, FPR={fpr}, Precision={precision}, Accuracy={accuracy}"
        )
        results[i] = (
            float32(lower_bound),
            float32(upper_bounds[i]),
            tpr,
            fpr,
            precision,
            accuracy,
        )
        LOGGER.debug(f"=============================================")

    return results


def bootstrap_indexes(
    indexes: NDArray[Shape["*"], UInt16],
    stratum_size: Optional[uint16] = None,
    strata_count: uint16 = uint16(100),
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
        stratum_size = uint16(n_population)
    if stratum_size > n_population:
        stratum_size = uint16(n_population)

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
    """ """

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
                strata_count=uint16(15),
                strata_size=uint16(30),
                boundary_width=BOUNDARY_WIDTH,
                jaccard_threshold=float32(0.20),
                max_workers=uint8(2),
                caching=False,
                gpu_caching=True,
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

    def _run_similarity_analysis(
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
        workers = self.config.max_workers

        LOGGER.debug(
            f">> Running similarity analysis with {workers} workers for {len(clist)} color tags."
        )

        # Get the datasets for cloud and sky images so we can reuse across Processes
        cloudset, skyset = get_datasets_vstack(camera)

        with ThreadPoolExecutor(max_workers=int(workers)) as executor:

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

    def _load_batch(
        self,
        camera: Camera,
        tag: ColourTag,
        bootstrap_samples: NDArray[Shape["*, *"], UInt16],
    ) -> tuple[tuple[int, int], GrayScaleImage, ColorImage]:
        """
        Load a batch of ground truth masks and reference images for the specified color tag.
        Create a composite image of the ground truth masks and a composite image of the reference images According to the indices in the bootstrap samples.
        Args:
            camera (Camera): Camera instance to use for analysis.
            tag (ColourTag): Color tag to analyze.
            bootstrap_samples (NDArray[Shape["*, *"], UInt16]): Indices of the samples to use for bootstrapping.
        Returns:
            tuple[tuple[int, int]], BitMapImage, ColorImage]: A tuple containing the composite ground truth masks, reference images, and shape (height, width).
        """

        sample_indices = bootstrap_samples[0]
        gt_sample, _ = get_masks_vstacks_sparse(camera, sample_indices[:1])
        ref_sample = get_reference_vstacks_sparse(camera, tag, sample_indices[:1])

        h, w = gt_sample.shape
        _, _, c = ref_sample.shape

        total_height = h * bootstrap_samples.shape[0] * bootstrap_samples.shape[1]

        all_gt_cpu = npzeros((total_height, w), dtype=uint8)
        all_ref_cpu = npzeros((total_height, w, c), dtype=uint8)

        current_row = 0

        for stratum_idx in range(bootstrap_samples.shape[0]):
            sample_indices = bootstrap_samples[stratum_idx]

            gt_masks, _ = get_masks_vstacks_sparse(camera, sample_indices)
            ref_imgs = get_reference_vstacks_sparse(camera, tag, sample_indices)

            if gt_masks is not None and ref_imgs is not None:
                gt_uint8 = (gt_masks * 255).astype(uint8)

                rows_to_add = gt_uint8.shape[0]
                all_gt_cpu[current_row : current_row + rows_to_add] = gt_uint8
                all_ref_cpu[current_row : current_row + rows_to_add] = ref_imgs
                current_row += rows_to_add

        return (total_height, w), all_gt_cpu, all_ref_cpu

    def _load_boundary_batch(
        self,
        index: uint8,
        camera: Camera,
        tag: ColourTag,
        bootstrap_samples: NDArray[Shape["*, *"], UInt16],
        boundaries: tuple[uint8, list[uint8]],
    ) -> tuple[tuple[int, int], GrayScaleImage, GrayScaleImage]:
        """
        Load a batch of ground truth masks and reference images for the specified color tag.
        We load the ground truth masks and reference images for the specified color tag, ina ccordance with the indices in the bootstrap samples.
        We create a composite image of the ground truth masks and a composite image of the reference images.
        We do this for each lower boundary in the boundaries, creating a larger composite image.
        """

        LOGGER.debug(
            f"load_boundary_batch :: {tag.tag} :: Loading batch for camera {camera.model.value} and color tag {tag.tag}"
        )

        dims, big_gt_mask, big_ref_mask = self._load_batch(
            camera, tag, bootstrap_samples
        )

        h, w = big_gt_mask.shape
        size = len(boundaries[1])
        height = h * size
        width = w

        bigger_bound_mask = npzeros((height, width), dtype=uint8)

        LOGGER.debug(
            f"load_boundary_batch :: {tag.tag} :: Ground truth mask is {big_gt_mask.shape}, Reference is {big_ref_mask.shape}"
        )

        current_row = 0
        lowerbound = boundaries[0]
        coi = cv2.split(big_ref_mask)[int(index)]
        for upperbound in boundaries[1]:

            boundmask = cv2.inRange(  # type: ignore
                coi,
                int(lowerbound),
                int(upperbound),
            )

            bigger_bound_mask[current_row : current_row + h] = boundmask
            current_row += h

        return dims, big_gt_mask, bigger_bound_mask

    def _analyze_channel_roc(
        self: Self,
        camera: Camera,
        tag: ColourTag,
        index: uint8,
        bootstrap_samples: NDArray[Shape["*, *"], UInt16],
        boundaries: BoundaryArray,
    ) -> NDArray[Shape["*, 6"], Float32]:

        LOGGER.debug(
            f"analayze_channel_roc :: Analyzing channel {tag.tag} index {index} for camera {camera.model.value}"
        )

        fullboundcount = 0
        for boundarygroup in boundaries:
            fullboundcount += len(boundarygroup[1])

        results = empty((fullboundcount, 6), dtype=float32)
        currentrow = 0

        for boundarygroup in boundaries:

            lowerbound, upperbounds = boundarygroup

            dims, gtmask, boundmask = self._load_boundary_batch(
                index=index,
                camera=camera,
                tag=tag,
                bootstrap_samples=bootstrap_samples,
                boundaries=boundarygroup,
            )

            LOGGER.debug(
                f"analayze_channel_roc :: {tag.tag} :: Loaded Boundary masks for bounds starting with {boundarygroup[0]}"
            )
            LOGGER.debug(
                f"analayze_channel_roc :: {tag.tag} Ground truth masks ::  dtype {gtmask.dtype}, Shape {gtmask.shape} || Boundary masks :: dtype {boundmask.dtype} Shape {boundmask.shape}"
            )

            gtumat = cv2.UMat(gtmask)  # type: ignore
            boundumat = cv2.UMat(boundmask)  # type: ignore

            LOGGER.debug(
                f"analayze_channel_roc :: Mats loaded to gpu :: Free memory :: {cv2.ocl.Device.getDefault().globalMemSize() // (1024*1024)} MB"
            )

            confusionresult = _batch_compute_confusion_matrix(
                lowerbound,
                upperbounds,
                dims,
                gtumat,
                boundumat,
            )

            del boundumat, gtumat

            LOGGER.debug(
                f"analayze_channel_roc :: {tag.tag} :: Computed confusion matrix for bounds starting with {boundarygroup[0]}"
            )

            results[currentrow : currentrow + len(upperbounds)] = confusionresult
            currentrow += len(upperbounds)

        LOGGER.debug(
            f"analayze_channel_roc :: Completed channel analysis for {tag.tag} index {index}"
        )
        return results

    def analyze_roc(
        self: Self, camera: Camera, colortags: list[ColourTag], overwrite: bool = False
    ) -> dict[str, NDArray[Shape["6"], Float32]]:

        LOGGER.info(f"Starting ROC analysis for camera {camera.model.value}")

        similarityresults = self._run_similarity_analysis(camera, colortags)
        if not similarityresults.size:
            LOGGER.warning("No similarity results found")
            return {}

        bootstrap_samples = bootstrap_indexes(
            array([i for i in range(len(camera.cloud_images_paths()))]),
            stratum_size=self.config.strata_size,
            strata_count=self.config.strata_count,
        )

        boundaries = generate_boundary_permutations(
            step_size=self.config.boundary_width
        )

        results = {}

        try:
            analysis_tasks = []
            for similarity_result in similarityresults:
                ctag_name = similarity_result["tag"]

                best_channel = similarity_result["components"][0]

                if best_channel["score"] > self.config.jaccard_threshold:
                    LOGGER.info(
                        f"Skipping {ctag_name} - similarity too high: {best_channel['score']}"
                    )
                    continue

                channel_index = best_channel["index"]
                tag = ColourTag.match(ctag_name)
                if tag is ColourTag.UNKNOWN:
                    LOGGER.warning(f"Unknown color tag: {ctag_name}, skipping analysis")
                    continue

                analysis_tasks.append(
                    {
                        "ctag_name": ctag_name,
                        "tag": tag,
                        "channel_index": channel_index,
                        "best_channel": best_channel,
                    }
                )

            if not analysis_tasks:
                LOGGER.warning("No valid analysis tasks found")
                return {}

            LOGGER.info(
                f"Starting ROC analysis for {len(analysis_tasks)} color channels"
            )

            with ThreadPoolExecutor(
                max_workers=int(self.config.max_workers)
            ) as executor:
                futures = {}
                for task in analysis_tasks:
                    future = executor.submit(
                        self._analyze_channel_roc,
                        camera,
                        task["tag"],
                        task["channel_index"],
                        bootstrap_samples,
                        boundaries,
                    )
                    futures[future] = task

                for future in as_completed(futures, timeout=600):
                    task = futures[future]
                    ctag_name = task["ctag_name"]
                    best_channel = task["best_channel"]

                    try:
                        channel_results = future.result()
                        results[f"{ctag_name}_{best_channel['component']}"] = (
                            channel_results
                        )
                        LOGGER.info(
                            f"Completed analysis for {ctag_name} channel {best_channel['component']}"
                        )

                    except Exception as e:
                        LOGGER.error(f"Failed to analyze {ctag_name}: {e}")

            LOGGER.info(f"ROC analysis completed for camera {camera.model.value}")
            return results

        except Exception as e:
            LOGGER.error(f"Error during ROC analysis: {e}")
            return {}
