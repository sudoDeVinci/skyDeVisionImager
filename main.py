from server.imageanalysis import (
    get_datasets_vstack,
    Camera,
    ColourTag,
    ColorSpaceAnalyzer
)

from server.db import CameraModel

model = CameraModel.DSLR
camera = Camera(model)

analyzer = ColorSpaceAnalyzer(camera)
results = analyzer.analyze_channel(ColourTag.HSV)
print(results)

