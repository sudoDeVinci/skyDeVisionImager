from server.imageanalysis import (
    get_datasets_vstack,
    Camera,
    ColourTag,
    ROCAnalyzer,
    BoundaryRange,
)
from cv2 import imread, imshow, imwrite, waitKey
from numpy import array, uint8
from server.db import CameraModel
from typing import no_type_check

@no_type_check
def main():
    model = CameraModel.DSLR
    ctag = ColourTag.HSV
    camera = Camera(model)

    analyzer = ROCAnalyzer(camera=camera, config=None)
    analyzer.run_similarity_analysis(
        camera=camera,
        ctags=[ctag]
    )

if __name__ == "__main__":
    main()



