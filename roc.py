from server.imageanalysis import (
    Camera,
    ColourTag,
    ROCAnalyzer,
    AnalysisConfiguration,
    verify_gpu_setup,
)
from cv2 import imshow, waitKey, ocl
from numpy import array, uint8, zeros, uint16, float32
from server.db import CameraModel
from typing import no_type_check
from json import dump
from logging import getLogger, Logger

ocl.setUseOpenCL(True)


@no_type_check
def roc():
    camera = Camera(CameraModel.DSLR)
    config = AnalysisConfiguration(
        strata_count=uint16(10),
        strata_size=uint16(10),
        boundary_width=uint8(5),
        jaccard_threshold=float32(0.15),
        max_workers=uint8(2),
        caching=False,
        gpu_caching=True,
    )
    analyzer = ROCAnalyzer(camera=camera, config=config)
    results = analyzer.analyze_roc(
        camera=camera,
        colortags=[ColourTag.HSV, ColourTag.RGB, ColourTag.YBR],
    )

    jsondict = {"config": config.to_dict(), "results": {}}
    jsondict["results"].update({tagname: [] for tagname in results.keys()})

    for ctag, metrics in results.items():
        print(f"Color Tag: {ctag}")
        for metric in metrics:
            lower, upper = int(metric[0]), int(metric[1])
            tpr, fpr, precision, accuracy = metric[2], metric[3], metric[4], metric[5]

            jsondict["results"][ctag].append(
                {
                    "lower": int(lower),
                    "upper": int(upper),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "precision": float(precision),
                    "accuracy": float(accuracy),
                }
            )

    with open("exampleroc.json", "w") as f:
        dump(jsondict, f, indent=4)


@no_type_check
def roc_load_batch():
    camera = Camera(CameraModel.DSLR)
    samples = array([[0, 1, 2, 3]], dtype=uint8)
    analyzer = ROCAnalyzer(camera=camera, config=None)

    gtmask, refimg = analyzer._load_batch(
        camera=camera, tag=ColourTag.RGB, bootstrap_samples=samples
    )

    imshow("Ground Truth Mask", gtmask)
    imshow("Reference Image", refimg)
    waitKey(0)


@no_type_check
def roc_load_batches():
    camera = Camera(CameraModel.DSLR)
    tag = ColourTag.RGB
    samples = array([[0, 1, 2, 3]], dtype=uint8)
    boundaries = (0, [i for i in range(0, 100, 5)])
    analyzer = ROCAnalyzer(camera=camera, config=None)
    gtmask, boundmask = analyzer._load_boundary_batch(
        index=0,
        camera=camera,
        tag=tag,
        bootstrap_samples=samples,
        boundaries=boundaries,
    )
    print(f"GT Mask Shape: {gtmask.shape}")
    print(f"Boundary Mask Shape: {boundmask.shape}")


if __name__ == "__main__":
    from cv2 import UMat

    verify_gpu_setup()

    roc()