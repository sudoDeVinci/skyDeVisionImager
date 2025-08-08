from server.imageanalysis import (
    Camera,
    ColourTag,
    ROCAnalyzer,
    get_datasets_vstacks_sparse,
    get_masks_vstacks_sparse,
    get_reference_vstacks_sparse,
    verify_gpu_setup,
)
from cv2 import imshow, waitKey, ocl
from numpy import array, uint8
from server.db import CameraModel
from typing import no_type_check
from json import dump

ocl.setUseOpenCL(True)


@no_type_check
def dataset_vstack():
    camera = Camera(CameraModel.DSLR)
    indices = [5, 4, 3, 2]
    cloudimg, _ = get_datasets_vstacks_sparse(camera=camera, indices=indices)
    cmask, _ = get_masks_vstacks_sparse(camera=camera, indices=indices)
    ref = get_reference_vstacks_sparse(
        camera=camera, ctag=ColourTag.RGB, indices=indices
    )
    imshow("Clouds", cloudimg)
    imshow("Cloud Masks", array(cmask * 255, dtype=uint8))
    imshow("Reference", ref)
    waitKey(0)


@no_type_check
def roc():
    camera = Camera(CameraModel.DSLR)
    analyzer = ROCAnalyzer(camera=camera, config=None)
    results = analyzer.analyze_roc(
        camera=camera,
        colortags=[ColourTag.HSV, ColourTag.RGB, ColourTag.YBR],
    )

    jsondict = {tagname: [] for tagname in results.keys()}

    for ctag, metrics in results.items():
        print(f"Color Tag: {ctag}")
        for metric in metrics:
            lower, upper = int(metric[0]), int(metric[1])
            tpr, fpr, precision, accuracy = metric[2], metric[3], metric[4], metric[5]

            jsondict[ctag].append(
                {
                    "lower": int(lower),
                    "upper": int(upper),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "precision": float(precision),
                    "accuracy": float(accuracy),
                }
            )

    with open("roc_results.json", "w") as f:
        dump(jsondict, f, indent=4)


def verify_opencl_usage():
    """Verify OpenCL is working and being used"""
    return verify_gpu_setup()


if __name__ == "__main__":
    if verify_opencl_usage():
        print("GPU setup verified successfully!")
    else:
        print("Warning: GPU setup issues detected")

    roc()
