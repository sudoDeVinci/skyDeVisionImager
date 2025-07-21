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
    max as npmax,
    min as npmin,
)

from typing import Optional
from plotly.graph_objects import Surface, Figure
from numba import njit
from scipy.interpolate import griddata
from scipy.ndimage import median_filter, gaussian_filter
from pathlib import Path

from nptyping import (
    NDArray,
    Shape,
    UInt8,
    Float32,
    Float64,
    Bool
)

from rasterio import open as rio_open, io # type: ignore

from io import TextIOWrapper


type GeoCoordinateArray = NDArray[Shape["*, 3"], Float32]


def tfw_to_xyz(
    tfw_file: TextIOWrapper, geotiff_file: io.DatasetReader
) -> GeoCoordinateArray:
    """
    Convert GeoTransform World File (TFW) along with a GeoTIFF (TIFF) file,
    to a format suitable for XYZ coordinate data processing.

    Args:
        tfw_file (TextIOWrapper | _WrappedBuffer): The TFW file containing geospatial transformation data.
        geotiff_file (io.DatasetReader): The GeoTIFF file containing topographic data.
    """

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


@njit
def mask_outliers_iqr(data: GeoCoordinateArray) -> NDArray[Shape["*"], Bool]:
    """
    Generate a boolean mask to identify outliers in the z-coordinate of an XYZ array using the IQR method.

    Args:
        data (GeoCoordinateArray): An array of XYZ coordinates.

    Returns:
        NDArray[Shape["*"], Bool]: A boolean mask where `True` indicates an inlier.
    """
    z_values = data[:, 2]
    q1 = percentile(z_values, 25)
    q3 = percentile(z_values, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (z_values >= lower_bound) & (z_values <= upper_bound)

    return mask



def meshify(
    xyz: GeoCoordinateArray,
    points: uint16 = 500,
    scalar: float32 = 1.0,
    interpolation: str = "linear",
    smooth_sigma: float32 = 1.0,
) -> NDArray[Shape["3, *"], Float32]:
    
    mask = mask_outliers_iqr(xyz)
    xyz = xyz[mask]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # Get aspect ratio for scalling grid
    x_range = npmax(x) - npmin(x)
    y_range = npmax(y) - npmin(y)
    z_range = npmax(z) - npmin(z)
    aspect_ratio = mean([x_range, y_range]) / z_range

    # Apply log scaling to z values to preserve detail
    z_log = log1p(z - npmin(z))
    z_normalized = (z_log - npmin(z_log)) / (npmax(z_log) - npmin(z_log))

    # Scale height with aspect ratio
    z_scaled = z_normalized * scalar * aspect_ratio

    # Check we have at least 4 points to form a mesh
    if len(xyz) < 4:
        raise ValueError("Not enough points to form a mesh. At least 4 points are required.")
    
    # Create a grid for interpolation
    x_grid = linspace(npmin(x), npmax(x), points, dtype=float32)
    y_grid = linspace(npmin(y), npmax(y), points, dtype=float32)
    x_mesh, y_mesh = meshgrid(x_grid, y_grid, copy=False)

    z_mesh = griddata(
        (x, y),
        z_scaled,
        (x_mesh, y_mesh),
        method=interpolation,
        rescale=True
    )
    
    # Denoising then Gaussian smoothing
    if isnan(z_mesh).all():
        raise ValueError("Interpolation resulted in NaN values. Check input data.")
    
    z_mesh = median_filter(z_mesh, size=3)
    z_mesh = gaussian_filter(z_mesh, sigma=smooth_sigma)

    return array([x_mesh, y_mesh, z_mesh], dtype=float32)


def graph_plotly(
    meshes: NDArray[Shape["3, *"], Float32],
    title: str = "Topography Mesh",
    cmapstr: str = "viridis",
    fig: Optional[Figure] = None
) -> Figure:
    x_mesh, y_mesh, z_mesh = meshes

    z_min  = nanmin(z_mesh)
    z_max  = nanmax(z_mesh)
    contour_size = (z_max - z_min) / 8 if not isnan(z_max - z_min) else 2.0

    # Add surface plot with simplified settings
    fig.add_trace(
        Surface(
            x=x_mesh,
            y=y_mesh,
            z=z_mesh,
            colorscale=cmapstr,
            opacity=0.85,
            showscale=False,
            hovertemplate="""x: %{x:.1f}
            <br>
            y: %{y:.1f}
            <br>
            height: %{z:.1f}
            <extra></extra>""",
            # Simplified contours with NaN handling
            contours={
                "z": {
                    "show": True,
                    "start": z_min,
                    "end": z_max,
                    "size": contour_size,
                    "width": 1,
                    "color": "white",
                    "usecolormap": True,
                    "highlightcolor": "white",
                    "highlightwidth": 1,
                }
            },
        )
    )

    # Update layout with cleaner styling
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 30},
        },
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
        scene={
            "xaxis": {
                "tickfont": {"size": 12},
                "showgrid": False,
                "showline": False,
                "linewidth": 1,
                "linecolor": "gray",
            },
            "yaxis": {
                "tickfont": {"size": 12},
                "showgrid": False,
                "showline": False,
                "linewidth": 1,
                "linecolor": "gray",
            },
            "zaxis": {
                "tickfont": {"size": 13},
                "showgrid": False,
                "showline": True,
                "linewidth": 2,
                "linecolor": "gray",
            },
            "bgcolor": "#f0f0f0",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
            "aspectratio": {"x": 1, "y": 1, "z": 0.7},
        },
        width=2000,
        height=2000,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    return fig


def graph_plotly(
    tfwfile: Path,
    tifffile: Path
) -> Figure:
    if not tifffile.exists():
        raise ValueError(f"GeoTIFF file {tifffile} does not exist.")
    if not tfwfile.exists():
        raise ValueError(f"TFW file {tfwfile} does not exist.")
    
    with open(tfwfile, "r") as tfw:
        with rio_open(tifffile) as tif:
            xyz = tfw_to_xyz(tfw, tif)

    meshes = meshify(xyz, points=500, scalar=1.0, interpolation="linear", smooth_sigma=1.0)
    fig = Figure()
    fig = graph_plotly(meshes, title="Topography Mesh", cmapstr="viridis", fig=fig)
    return fig