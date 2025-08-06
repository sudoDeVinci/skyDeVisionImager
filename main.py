from server.imageanalysis import (
    get_datasets_vstack,
    Camera,
    ColourTag,
    ROCAnalyzer,
    BoundaryRange,
    get_datasets_vstacks_sparse,
    get_masks_vstacks_sparse,
    get_reference_vstacks_sparse,
)
from cv2 import imread, imshow, imwrite, waitKey
from numpy import array, uint8
from server.db import CameraModel
from typing import no_type_check


@no_type_check
def jaccard():
    model = CameraModel.DSLR
    ctag = ColourTag.HSV
    camera = Camera(model)

    analyzer = ROCAnalyzer(camera=camera, config=None)
    analyzer.run_similarity_analysis(camera=camera, ctags=[ColourTag.HSV, ColourTag.RGB, ColourTag.YBR])

@no_type_check
def dataset_vstack():
    camera = Camera(CameraModel.DSLR)
    indices = [5,4,3,2]
    cloudimg, _ = get_datasets_vstacks_sparse(camera=camera, indices=indices)
    cmask, _ = get_masks_vstacks_sparse(camera=camera, indices=indices)
    ref = get_reference_vstacks_sparse(camera=camera, indices=indices)
    imshow("Clouds", cloudimg)
    imshow("Cloud Masks", array(cmask*255, dtype=uint8))
    imshow("Reference", ref)
    waitKey(0)

@no_type_check
def roc():
    camera = Camera(CameraModel.DSLR)
    analyzer = ROCAnalyzer(camera=camera, config=None)
    results = analyzer.analyze_roc(
        camera=camera,
        colortags=[ColourTag.HSV, ColourTag.RGB]
    )
    print(results)

if __name__ == "__main__":
    roc()
    #dataset_vstack()