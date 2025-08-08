from __future__ import annotations
from typing import Callable, NamedTuple
from enum import Enum
from numpy import (
    vstack,
    percentile,
    asarray,
    uint8,
    bool_,
    array,
    empty,
    column_stack,
    where,
    unique,
    zeros as npzeros,
)
from pathlib import Path
from cv2 import (
    COLOR_BGR2RGB,
    COLOR_BGR2HSV,
    COLOR_BGR2YCrCb,
    IMREAD_GRAYSCALE,
    cvtColor,
    imread,
)
from .configuration import Camera

from cv2.typing import MatLike
from typing import Final, Optional, Any, cast, Annotated
from nptyping import NDArray, Shape, UInt8, Bool, UInt16


type ColorImage = Annotated[
    NDArray[Shape["Height, Width, PixelTriplet"], UInt8],
    "Represents a 3-channel image (H, W, 3)",
]
"""
Represents a 3-channel image (H, W, 3)"]
"""

type ColorImageArray = Annotated[
    NDArray[Shape["*, Height, Width, PixelTriplet"], UInt8],
    "Represents a batch of 3-channel images (N, H, W, 3)",
]
"""
Represents a batch of 3-channel images (N, H, W, 3)"]
"""

type GrayScaleImage = Annotated[
    NDArray[Shape["Height, Width"], UInt8],
    "Represents a single-channel grayscale image (H, W)",
]
"""
Represents a single-channel grayscale image (H, W)"]
"""

type BitMapImage = Annotated[
    NDArray[Shape["*, *"], Bool],
    "Represents a single-channel bitmap image (H, W)",
]
"""
Represents a single-channel bitmap image (H, W)"]
"""

type PixelData = Annotated[
    NDArray[Shape["*, 3"], UInt8],
    "Represents a list of pixels, each with 3 channels (N, 3)",
]
"""
Represents a list of pixels, each with 3 channels (N, 3)"]
"""

type ComponentData = Annotated[
    NDArray[Shape["*"], UInt8], "Represents a single channel of pixel data (N,)"
]
"""
Represents a single channel of pixel data (N,)"]
"""

type FrequencyTable = Annotated[
    NDArray[Shape["*, 2"], UInt8], "Represents a frequency table (UniqueValues, 2)"
]
"""
Represents a frequency table (UniqueValues, 2)"]
"""

type DecomposedData = Annotated[
    NDArray[Shape["*"], Any], "Represents an array of frequency tables"
]
"""
Represents an array of frequency tables"]
"""

type ChannelData = Annotated[
    NDArray[Shape["*"], UInt8], "Represents a single channel of pixel data (N,)"
]
"""
Represents a single channel of pixel data (N,)"]
"""

IMG_EXTENSIONS: Final[tuple[str, ...]] = ("jpg", "png", "jpeg", "bmp", "svg")


class ColourSpaceInfo(NamedTuple):
    """Information about a color space including its components and processing details."""

    components: tuple[str, ...]
    tag: str
    cv2alias: int
    callback: Callable[[ColorImage], PixelData]


def _is_img(filename: str) -> bool:
    """Check if the file is an image based on its extension."""
    return filename.lower().endswith(IMG_EXTENSIONS)


def extract_nonblack_RGB(image: ColorImage) -> PixelData:
    """
    Extract the non-black pixels from a colour-masked image in RGB format.
    Non-black is defined as any pixel where at least one channel is > 0.
    """
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    non_black_indices = where((red > 0) | (green > 0) | (blue > 0))
    non_black_data = column_stack(
        (red[non_black_indices], green[non_black_indices], blue[non_black_indices])
    )
    return non_black_data


def extract_nonblack_HSV(image: ColorImage) -> PixelData:
    """
    Extract the non-black pixels from a colour-masked image in HSV format.
    In HSV, black is defined as Value (brightness) = 0, regardless of Hue and Saturation.
    For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]
    """
    h, s, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    non_black_indices = where(v > 0)
    non_black_data = column_stack(
        (h[non_black_indices], s[non_black_indices], v[non_black_indices])
    )
    return non_black_data


def extract_nonblack_YBR(image: ColorImage) -> PixelData:
    """
    Extract the non-black pixels from a colour-masked image in YCrCb format.
    In YCrCb, black is primarily defined by Y (luminance) = 0.
    For consistency with other functions, we check Y > 0.
    """
    Y, Cr, Cb = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    non_black_indices = where(Y > 0)
    non_black_data = column_stack(
        (Y[non_black_indices], Cr[non_black_indices], Cb[non_black_indices])
    )
    return non_black_data


class ColourTag(Enum):
    """
    Enum for different colour tags used in image processing.
    Each tag has associated components, a string identifier, an OpenCV alias,
    and a callback function for processing images.
    """

    RGB = ColourSpaceInfo(
        components=("red", "green", "blue"),
        tag="RGB",
        cv2alias=COLOR_BGR2RGB,
        callback=extract_nonblack_RGB,
    )

    HSV = ColourSpaceInfo(
        components=("hue", "saturation", "value"),
        tag="HSV",
        cv2alias=COLOR_BGR2HSV,
        callback=extract_nonblack_HSV,
    )

    YBR = ColourSpaceInfo(
        components=("Y", "Cr", "Cb"),
        tag="YCrCb",
        cv2alias=COLOR_BGR2YCrCb,
        callback=extract_nonblack_YBR,
    )

    UNKNOWN = ColourSpaceInfo(
        components=(),
        tag="UNKNOWN",
        cv2alias=-1,
        callback=lambda image: empty((0, 3), dtype=uint8),
    )

    @classmethod
    def match(cls, tag: str) -> ColourTag:
        """
        Get the ColourTag enum member from a string.
        If the string does not match any tag, return UNKNOWN.
        """
        for member in cls:
            if member.value.tag.lower() == tag.lower():
                return member
        return cls.UNKNOWN

    @property
    def components(self) -> tuple[str, ...]:
        """Get the color components for this color space."""
        return self.value.components

    @property
    def tag(self) -> str:
        """Get the tag string for this color space."""
        return self.value.tag

    @property
    def cv2alias(self) -> int:
        """Get the OpenCV constant for this color space."""
        return self.value.cv2alias

    @property
    def callback(self) -> Callable[[ColorImage], PixelData]:
        """Get the processing callback for this color space."""
        return self.value.callback


def get_nonblack_pixels(image: Optional[ColorImage], ctag: ColourTag) -> PixelData:
    """
    Extract non-black pixels from an image file based on the specified color tag.

    Args:
        img (ColorImage | None): The image data to process.
        ctag (ColourTag): The color tag specifying the color space.

    Returns:
        PixelData: Array of non-black pixel data in the specified color space.

    Raises:
        ValueError: If the file is not a valid image, or if it cannot be read as an image.
    """
    if image is None:
        raise ValueError(
            f"ColorImage data is None. Cannot extract non-black pixels for tag '{ctag.tag}'."
        )

    converted = cvtColor(cast(MatLike, image), ctag.cv2alias)

    nonblack = ctag.callback(cast(ColorImage, converted))

    return nonblack


def remove_outliers_iqr(data: PixelData) -> PixelData:
    """
    Remove outliers from data using IQR.
    Data points that fall below Q1 - 1.5 IQR or above the third quartile
    Q3 + 1.5 IQR are outliers.

    Args:
        data (PixelData): The input data array, expected to be 2D.

    Returns:
        PixelData: The data array with outliers removed, preserving the 2D structure.
    """
    Q1 = percentile(data, 25, axis=0)  # Calculate Q1 along columns (axis=0)
    Q3 = percentile(data, 75, axis=0)  # Calculate Q3 along columns (axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Ensure 2D structure is preserved
    return data[((data >= lower_bound) & (data <= upper_bound)).all(axis=1)]


def count(xyz_sk: ComponentData) -> FrequencyTable:
    """
    Return a frequency table of the integers in an input array

    Args:
        xyz_sk (ComponentData): Input array of integers.

    Returns:
        FrequencyTable: A 2D array where the first column contains unique integers
                        and the second column contains their corresponding counts.
    """
    uni, counts = unique(xyz_sk, return_counts=True)
    freq = asarray((uni, counts), dtype=uint8, order="C").T
    return freq


def decompose(data: PixelData, n_components: int) -> DecomposedData:
    """
    Decompose the data into n component frequency tables.
    Args:
        data (PixelData): Input data array.
        n_components (int): Number of components to decompose into.

    Returns:
        DecomposedData: A 1D array where each element is a frequency table for the corresponding component.
    """
    out = [count(array(data[:, i], dtype=uint8)) for i in range(n_components)]
    return array(out, dtype=object)


def frequency_distribution(image: ColorImage, ctag: ColourTag) -> DecomposedData:
    data = remove_outliers_iqr(get_nonblack_pixels(image, ctag))
    return decompose(data, len(ctag.components))


def get_datasets_vstack(
    camera: Camera,
) -> tuple[ColorImage, ColorImage]:
    """
    Get the sky and cloud ground truth datasets for a camera.
    The images are stitched together from the camera's sky and cloud directories.
    Args:
        camera (Camera): The camera instance to get datasets for.
    Returns:
        tuple[ColorImage, ColorImage]: A tuple containing the cloud images and sky images - stitched together vertically.
    """

    cloud_paths = list(camera.cloud_images_paths())
    sky_paths = list(camera.sky_images_paths())

    # Read first image to get dimensions
    sample = imread(str(cloud_paths[0]))
    h, w, c = sample.shape

    # Pre-allocate final arrays
    clouds = empty((h * len(cloud_paths), w, c), dtype=uint8)
    skies = empty((h * len(sky_paths), w, c), dtype=uint8)

    # Fill arrays directly (no intermediate list)
    for i, path in enumerate(cloud_paths):
        img = imread(str(path))
        clouds[i * h : (i + 1) * h] = img

    for i, path in enumerate(sky_paths):
        img = imread(str(path))
        skies[i * h : (i + 1) * h] = img

    return clouds, skies


def get_datasets_vstacks_sparse(
    camera: Camera, indices: NDArray[Shape["*"], UInt16]
) -> tuple[ColorImage, ColorImage]:
    """
    Create sparse stacked images where output position matches index value.
    """
    cloud_paths = list(camera.cloud_images_paths())
    sky_paths = list(camera.sky_images_paths())

    # Use first valid index for dimensions
    sample_idx = int(indices[0])
    sample = imread(str(cloud_paths[sample_idx]))
    h, w, c = sample.shape

    # Pre-allocate for actual number of images
    clouds = npzeros((h * len(indices), w, c), dtype=uint8)
    skies = npzeros((h * len(indices), w, c), dtype=uint8)

    # Fill arrays sequentially
    for i, idx in enumerate(indices):
        idx = int(idx)
        if idx < len(cloud_paths) and idx < len(sky_paths):
            img_cloud = imread(str(cloud_paths[idx]))
            img_sky = imread(str(sky_paths[idx]))
            clouds[i * h : (i + 1) * h] = img_cloud
            skies[i * h : (i + 1) * h] = img_sky

    return clouds, skies


def get_masks_vstack(
    camera: Camera,
) -> tuple[BitMapImage, BitMapImage]:
    """
    Get the sky and cloud masks for a camera.
    The masks are stitched together from the camera's sky and cloud mask directories.
    Args:
        camera (Camera): The camera instance to get masks for.
    Returns:
        tuple[ColorImage, ColorImage]: A tuple containing the cloud masks and sky masks - stitched together vertically.
    """

    cloud_masks = [imread(str(file)) for file in camera.cloud_masks_paths()]
    sky_masks = [imread(str(file)) for file in camera.sky_masks_paths()]

    clouds = vstack(cloud_masks, dtype=bool_)
    skies = vstack(sky_masks, dtype=bool_)

    return clouds, skies


def get_masks_vstacks_sparse(
    camera: Camera, indices: NDArray[Shape["*"], UInt16]
) -> tuple[BitMapImage, BitMapImage]:
    from cv2 import imshow, waitKey

    """
    Create sparse stacked masks where output position matches index value.
    """
    cloud_mask_paths = list(camera.cloud_masks_paths())
    sky_mask_paths = list(camera.sky_masks_paths())

    # Use first valid index for dimensions
    sample_idx = int(indices[0])
    sample = imread(str(cloud_mask_paths[sample_idx]), IMREAD_GRAYSCALE)
    h, w = sample.shape

    # Pre-allocate for actual number of images
    clouds = npzeros((h * len(indices), w), dtype=bool_)
    skies = npzeros((h * len(indices), w), dtype=bool_)

    # Fill arrays sequentially
    for i, idx in enumerate(indices):
        idx = int(idx)
        if idx < len(cloud_mask_paths) and idx < len(sky_mask_paths):
            img_cloud = imread(str(cloud_mask_paths[idx]), IMREAD_GRAYSCALE)
            img_sky = imread(str(sky_mask_paths[idx]), IMREAD_GRAYSCALE)
            clouds[i * h : (i + 1) * h] = img_cloud > 0
            skies[i * h : (i + 1) * h] = img_sky > 0

    return clouds, skies


def get_reference_vstacks_sparse(
    camera: Camera, ctag: ColourTag, indices: NDArray[Shape["*"], UInt16]
) -> ColorImage:
    """
    Get sparse stacked images for reference datasets.
    """
    refpaths = list(camera.reference_images_paths())

    # Use first valid index for dimensions
    sample_idx = int(indices[0])
    sample = imread(str(refpaths[sample_idx]))
    h, w, c = sample.shape

    # Pre-allocate for actual number of images
    refimgs = npzeros((h * len(indices), w, c), dtype=uint8)

    # Fill arrays sequentially
    for i, idx in enumerate(indices):
        idx = int(idx)
        if idx < len(refpaths):
            img = imread(str(refpaths[idx]))
            refimgs[i * h : (i + 1) * h] = img

    # Convert to the specified color space
    refimg = cvtColor(cast(MatLike, refimgs), ctag.cv2alias)

    return cast(ColorImage, refimg)
