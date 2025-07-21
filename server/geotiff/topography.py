from numpy import (
    array,
    arange,
    linspace,
    meshgrid,
    nanmin,
    nanmax,
    isnan,
    zeros,
    float32,
    float64,
    uint16,
    percentile,
    mean,
    log1p,
    max as npmax,
    min as npmin,
)

from typing import Optional
from plotly.graph_objects import Surface, Figure  # type: ignore
from numba import njit, types, prange  # type: ignore
from scipy.interpolate import griddata  # type: ignore
from scipy.ndimage import median_filter, gaussian_filter  # type: ignore
from pathlib import Path

from nptyping import NDArray, Shape, Float32, Bool

from rasterio import open as rio_open, io  # type: ignore

import pickle
from hashlib import md5
from io import TextIOWrapper


type GeoCoordinateArray = NDArray[Shape["*, 3"], Float32]


@njit(
    types.npytypes.Array(types.float32, 2, "C")(
        types.npytypes.Array(types.float32, 2, "C"),
        types.npytypes.Array(types.float32, 2, "C"),
        types.npytypes.Array(types.float32, 2, "C"),
    ),
    fastmath=True,
    cache=True,
)
def _vectorized_xyz_extraction(
    longitudes, latitudes, raster_data
) -> GeoCoordinateArray:
    """Numba-optimized function to extract valid XYZ coordinates."""
    print(
        f"Longitudes shape ::: W:{longitudes.shape[0]}, H:{longitudes.shape[1]},"
        f"Latitudes shape ::: W:{latitudes.shape[0]}, H:{latitudes.shape[1]}, "
        f"raster_data shape ::: W:{raster_data.shape[0]}, H:{raster_data.shape[1]}"
    )
    height, width = raster_data.shape

    # Pre-count valid points to avoid dynamic array resizing
    valid_count = 0
    for y in range(height):
        for x in range(width):
            if not isnan(raster_data[y, x]):
                valid_count += 1

    # Pre-allocate result array
    xyz = array([[0.0, 0.0, 0.0]] * valid_count, dtype=float32)
    # xyz = zeros()

    idx = 0
    for y in range(height):
        for x in range(width):
            if not isnan(raster_data[y, x]):
                xyz[idx, 0] = longitudes[y, x]
                xyz[idx, 1] = latitudes[y, x]
                xyz[idx, 2] = raster_data[y, x]
                idx += 1

    return xyz


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

    # We use copy as True here so we can get contiguous arrays later  - hoping for numba speedup
    height, width = int(geotiff_file.height), int(geotiff_file.width)
    x_indices, y_indices = meshgrid(
        arange(width, dtype=float32), arange(height, dtype=float32), copy=True
    )

    longitudes = a * x_indices + b * y_indices + e
    latitudes = c * x_indices + d * y_indices + f

    raster_data = array(geotiff_file.read(1), dtype=float32)

    # Use vectorized approach instead of list comprehension
    xyz = _vectorized_xyz_extraction(longitudes, latitudes, raster_data)

    return xyz


@njit(
    types.npytypes.Array(types.boolean, 1, "C")(
        types.npytypes.Array(types.float32, 2, "C")
    ),
    fastmath=True,
    cache=True,
)
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
    points: uint16 = uint16(500),
    scalar: float32 = float32(1.0),
    interpolation: str = "linear",
    smooth_sigma: float32 = float32(1.0),
) -> NDArray[Shape["3, *"], Float32]:

    # Check we have at least 4 points to form a mesh
    if len(xyz) < 4:
        raise ValueError(
            "Not enough points to form a mesh. At least 4 points are required."
        )

    print(
        f"Creating mesh with {len(xyz)} points, resolution: {points}, interpolation: {interpolation}, smooth_sigma: {smooth_sigma}"
    )

    @njit(
        types.Tuple(
            (
                types.npytypes.Array(types.float32, 1, "C"),
                types.npytypes.Array(types.float32, 1, "C"),
                types.npytypes.Array(types.float32, 1, "C"),
                types.npytypes.Array(types.float32, 1, "A"),
                types.npytypes.Array(types.float32, 1, "A"),
            )
        )(types.npytypes.Array(types.float32, 2, "C"), types.uint16),
        fastmath=False,
        parallel=True,
        cache=True,
    )
    def data_prepare(
        xyz: GeoCoordinateArray, points: uint16
    ) -> tuple[NDArray[Shape["*"], Float32], ...]:
        mask = mask_outliers_iqr(xyz)
        xyz = xyz[mask]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Get min/max values once
        x_min, x_max = npmin(x), npmax(x)
        y_min, y_max = npmin(y), npmax(y)
        z_min, z_max = npmin(z), npmax(z)

        # Get aspect ratio for scaling grid
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        aspect_ratio = mean(array([x_range, y_range], dtype=float32)) / (
            z_range + 1e-15
        )

        # Apply log scaling to z values to preserve detail
        z_shifted = z - z_min
        z_log = log1p(z_shifted)
        z_log_min, z_log_max = npmin(z_log), npmax(z_log)
        z_normalized = (z_log - z_log_min) / (z_log_max - z_log_min + 1e-15)

        # Scale height with aspect ratio
        z_scaled = z_normalized * scalar * aspect_ratio

        # Create a grid for interpolation
        x_grid = linspace(x_min, x_max, points)
        y_grid = linspace(y_min, y_max, points)

        return (
            x_grid.astype(float32),
            y_grid.astype(float32),
            z_scaled.astype(float32),
            x,
            y,
        )

    x_grid, y_grid, z_grid, x, y = data_prepare(xyz, points)

    x_mesh, y_mesh = meshgrid(x_grid, y_grid, copy=False)

    z_mesh = griddata(
        (x, y), z_grid, (x_mesh, y_mesh), method=interpolation, rescale=True
    )

    # Denoising then Gaussian smoothing
    if isnan(z_mesh).all():
        raise ValueError("Interpolation resulted in NaN values. Check input data.")

    # Apply filters only if smoothing is requested
    if smooth_sigma > 0:
        z_mesh = median_filter(z_mesh, size=3)
        z_mesh = gaussian_filter(z_mesh, sigma=smooth_sigma)

    return array([x_mesh, y_mesh, z_mesh], dtype=float64)


def plotly_surface(
    meshes: NDArray[Shape["3, *"], Float32],
    title: str = "Topography Mesh",
    cmapstr: str = "viridis",
    fig: Optional[Figure] = None,
) -> Figure:
    x_mesh, y_mesh, z_mesh = meshes

    z_min = nanmin(z_mesh)
    z_max = nanmax(z_mesh)
    contour_size = (z_max - z_min) / 8 if not isnan(z_max - z_min) else 2.0

    # Add surface plot with simplified settings
    if fig is None:
        fig = Figure()
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


def downsample_xyz(xyz: GeoCoordinateArray, factor: uint16) -> GeoCoordinateArray:
    """
    Downsample XYZ data by taking every nth point.

    Args:
        xyz: Input XYZ coordinate array
        factor: Downsampling factor (e.g., 2 = take every 2nd point)

    Returns:
        Downsampled XYZ array
    """
    if factor <= 1:
        return xyz

    n_points = len(xyz) // factor
    downsampled = array([[0.0, 0.0, 0.0]] * n_points, dtype=float32)

    for i in range(n_points):
        downsampled[i] = xyz[i * factor]

    return downsampled


def preprocess_data(
    xyz: GeoCoordinateArray, downsample_factor: int = 1, remove_outliers: bool = True
) -> GeoCoordinateArray:
    """
    Preprocess XYZ data with optional downsampling and outlier removal.

    Args:
        xyz: Input XYZ coordinate array
        downsample_factor: Factor by which to downsample data (1 = no downsampling)
        remove_outliers: Whether to remove outliers using IQR method

    Returns:
        Preprocessed XYZ array
    """
    # Remove outliers first if requested
    if remove_outliers:
        mask = mask_outliers_iqr(xyz)
        xyz = xyz[mask]

    # Downsample if factor > 1
    if downsample_factor > 1:
        xyz = downsample_xyz(xyz, uint16(downsample_factor))

    return xyz


def graph_plotly(
    tfwfile: Path,
    tifffile: Path,
    cachedir: Path = Path(__file__).parent / "__cache__",
    points: uint16 = uint16(200),
    use_cache: bool = True,
    downsample_factor: int = 1,
) -> Figure:

    if not tifffile.exists():
        raise ValueError(f"GeoTIFF file {tifffile} does not exist.")
    if not tfwfile.exists():
        raise ValueError(f"TFW file {tfwfile} does not exist.")
    if not cachedir.exists():
        cachedir.mkdir(exist_ok=True)

    # Create cache key based on file paths and parameters
    cache_key = md5(
        f"{tfwfile}_{tifffile}_{points}_{downsample_factor}".encode()
    ).hexdigest()
    cache_file = cachedir / f"mesh_{cache_key}.pkl"

    # Try to load from cache
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                meshes = pickle.load(f)
        except Exception:
            # If cache loading fails, regenerate
            meshes = None
    else:
        meshes = None

    # Generate mesh if not cached
    if meshes is None:
        with open(tfwfile, "r") as tfw:
            with rio_open(tifffile) as tif:
                # Optional downsampling for very large datasets
                if downsample_factor > 1:
                    # Read at reduced resolution
                    height = tif.height // downsample_factor
                    width = tif.width // downsample_factor
                    raster_data = tif.read(
                        1,
                        out_shape=(height, width),
                        resampling=1,  # Nearest neighbor for speed
                    )
                    # Update geotiff reader dimensions temporarily
                    original_height, original_width = tif.height, tif.width
                    tif._height, tif._width = height, width

                xyz = tfw_to_xyz(tfw, tif)
                print(f"Extracted {len(xyz)} XYZ points from GeoTIFF.")
                # Restore original dimensions
                if downsample_factor > 1:
                    tif._height, tif._width = original_height, original_width

        meshes = meshify(
            xyz,
            points=points,
            scalar=float32(1.0),
            interpolation="nearest",
            smooth_sigma=float32(1.0),
        )

        # Save to cache
        if use_cache:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(meshes, f)
            except Exception as e:
                print(f"Warning: Could not save to cache: {e}")

    fig = Figure()
    fig = plotly_surface(meshes, title="Topography Mesh", cmapstr="viridis", fig=fig)
    return fig
