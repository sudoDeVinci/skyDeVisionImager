from server.imageanalysis import (
    get_datasets_vstack,
    Camera,
    ColourTag,
    ColorSpaceAnalyzer,
    ROCAnalyzer,
    BoundaryRange
)
from cv2 import (
    imread,
    imshow,
    imwrite,
    waitKey
)
from numpy import array, uint8
from server.db import CameraModel

model = CameraModel.DSLR
ctag = ColourTag.HSV
camera = Camera(model)

redbound1 = BoundaryRange(10, 0)        #type: ignore
redbound2 = BoundaryRange(180, 170)     #type: ignore
img = imread(camera.blocked_images_paths()[0])

analyzer = ROCAnalyzer(
    config=None,
    camera=camera
)

threshed = analyzer.threshold(
    image=img,
    channelindex=0,
    boundary=redbound1  
)

threshed = array(threshed, dtype=uint8) * 255

imwrite("Thresholded.png", threshed)