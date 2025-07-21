from server.geotiff.topography import (
    graph_plotly
)
from pathlib import Path

BASEPATH = Path(__file__).parent / "hojddata" / "63_3"
TFWFILE = BASEPATH / "639_31_7550_2019.tfw"
TIFFFILE = BASEPATH / "639_31_7550_2019.tif"

figure = graph_plotly(
    tfwfile=TFWFILE,
    tifffile=TIFFFILE
)

figure.write_html(
    "topo.html",
    full_html=True,
    include_plotlyjs="True"
)

print("HTML file generated successfully: topo.html")
