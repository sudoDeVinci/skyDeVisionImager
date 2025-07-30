from server.geotiff import graph_plotly
from pathlib import Path
import time

BASEPATH = Path(__file__).parent / "hojddata" / "63_3"
TFWFILE = BASEPATH / "639_31_7550_2019.tfw"
TIFFFILE = BASEPATH / "639_31_7550_2019.tif"

start_time = time.time()

figure = graph_plotly(
    tfwfile=TFWFILE,
    tifffile=TIFFFILE,
    points=200,             # type: ignore
    use_cache=True,         # Enable caching
    downsample_factor=1,    # No downsampling
)

processing_time = time.time() - start_time
print(f"Processing completed in {processing_time:.2f} seconds")

figure.write_html("topo.html", full_html=True, include_plotlyjs="True")

print("HTML file generated successfully: topo.html")
