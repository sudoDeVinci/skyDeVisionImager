from numpy import (
    array,
    arange,
    linspace,
    meshgrid,
    nanmin,
    nanmax,
    isnan,
    unique,
    float32,
    float64,
    float16,
    uint16,
    percentile,
    mean,
    log1p,
)

from nptyping import (
    NDArray,
    Shape,
    UInt8,
    Float32,
    Float64,
)

from rasterio import open as rio_open, io  # type: ignore

from io import TextIOWrapper, _WrappedBuffer


type GeoCoordinateArray = NDArray[Shape["*, 3"], Float32]


def tfw_to_xyz(
    tfw_file: TextIOWrapper[_WrappedBuffer], geotiff_file: io.DatasetReader
) -> GeoCoordinateArray:
    """
    Convert GeoTransform World File (TFW) along with a GeoTIFF (TIFF) file,
    to a format suitable for XYZ coordinate data processing.

    Args:
        tfw_file (TextIOWrapper | _WrappedBuffer): The TFW file containing geospatial transformation data.
        geotiff_file (io.DatasetReader): The GeoTIFF file containing topographic data.
    """

    with rio_open(geotiff_file) as src:
        if not src.is_tiled:
            raise ValueError("GeoTIFF file must be tiled for this operation.")

    tfw_params = tuple([float32(line.strip()) for line in tfw_file.readlines()])
    a, b, c, d, e, f = tfw_params

    height, width = int(geotiff_file.height), int(geotiff_file.width)
    x_indices, y_indices = meshgrid(
        arange(width, dtype=float32), arange(height, dtype=float32)
    )

    longitudes = a * x_indices + b * y_indices + e
    latitudes = c * x_indices + d * y_indices + f

    raster_data = array(geotiff_file.read(1), dtype=float32)

    xyz = array(
        [
            [longitudes[y, x], latitudes[y, x], raster_data[y, x]]
            for y in range(height)
            for x in range(width)
            if not isnan(raster_data[y, x])
        ],
        dtype=float32,
    )

    return xyz
