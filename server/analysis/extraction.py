from __future__ import annotations
from typing import Callable, NamedTuple
from enum import Enum
from numpy import (
    vstack,
    percentile,
    asarray,
    uint8,
    array,
    empty,
    column_stack,
    ndarray,
    where,
    unique,
    save as npsave,
)
from numpy.typing import NDArray
from pathlib import Path
from cv2 import (
    COLOR_BGR2RGB,
    COLOR_BGR2HSV,
    COLOR_BGR2YCrCb,
    cvtColor,
    imread,
)

from cv2.typing import MatLike

from typing import Tuple, Union, Final, Optional, Any, cast

from .configuration import CameraModel, Camera

IMG_EXTENSIONS: Final[tuple[str, ...]] = ("jpg", "png", "jpeg", "bmp", "svg")

GenericImage = NDArray[uint8]
"""
A generic image type, typically a 2D or 3D array of pixel values.
It can represent grayscale or color images.
"""


class ColourSpaceInfo(NamedTuple):
    """Information about a color space including its components and processing details."""

    components: tuple[str, ...]
    tag: str
    cv2alias: int
    callback: Callable[[MatLike], MatLike]


def _is_img(filename: str) -> bool:
    """Check if the file is an image based on its extension."""
    return filename.lower().endswith(IMG_EXTENSIONS)


def extract_nonblack_RGB(image: MatLike) -> MatLike:
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


def extract_nonblack_HSV(image: MatLike) -> MatLike:
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


def extract_nonblack_YBR(image: MatLike) -> MatLike:
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
    def callback(self) -> Callable[[MatLike], MatLike]:
        """Get the processing callback for this color space."""
        return self.value.callback


def get_nonblack_pixels(image: Optional[MatLike], ctag: ColourTag) -> MatLike:
    """
    Extract non-black pixels from an image file based on the specified color tag.

    Args:
        img (GenericImage | None): The image data to process.
        ctag (ColourTag): The color tag specifying the color space.

    Returns:
        GenericImage: Array of non-black pixel data in the specified color space.

    Raises:
        ValueError: If the file is not a valid image, or if it cannot be read as an image.
    """
    if image is None:
        raise ValueError(
            f"Image data is None. Cannot extract non-black pixels for tag '{ctag.tag}'."
        )

    image = cvtColor(image, ctag.cv2alias)

    nonblack = ctag.callback(image)

    return nonblack


def remove_outliers_iqr(data: GenericImage) -> GenericImage:
    """
    Remove outliers from data using IQR.
    Data points that fall below Q1 - 1.5 IQR or above the third quartile
    Q3 + 1.5 IQR are outliers.

    Args:
        data (GenericImage): The input data array, expected to be 2D.

    Returns:
        GenericImage: The data array with outliers removed, preserving the 2D structure.
    """
    Q1 = percentile(data, 25, axis=0)  # Calculate Q1 along columns (axis=0)
    Q3 = percentile(data, 75, axis=0)  # Calculate Q3 along columns (axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Ensure 2D structure is preserved
    return data[((data >= lower_bound) & (data <= upper_bound)).all(axis=1)]


def count(xyz_sk: GenericImage) -> GenericImage:
    """
    Return a frequency table of the integers in an input array

    Args:
        xyz_sk (GenericImage): Input array of integers.

    Returns:
        GenericImage: A 2D array where the first column contains unique integers
                        and the second column contains their corresponding counts.
    """
    uni, counts = unique(xyz_sk, return_counts=True)
    freq = asarray((uni, counts), dtype=uint8).T
    return freq


def decompose(data: GenericImage, n_components: int):
    """
    Decompose the data into n component frequency tables.
    Args:
        data (GenericImage): Input data array.
        n_components (int): Number of components to decompose into.

    Returns:
        NDArray[object]: A 3D array where each element is a frequency table for the corresponding component.
    """
    out = [count(array(data[:, i], dtype=uint8)) for i in range(n_components)]
    return array(out, dtype=object)


def frequency_distribution(image: MatLike, ctag: ColourTag) -> NDArray:
    data = remove_outliers_iqr(cast(GenericImage, get_nonblack_pixels(image, ctag)))
    return decompose(data, len(ctag.components))
