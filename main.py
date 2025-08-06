from server.imageanalysis import (
    get_datasets_vstack,
    Camera,
    ColourTag,
    ROCAnalyzer,
    BoundaryRange,
    get_datasets_vstacks_sparse,
    get_masks_vstacks_sparse,
    get_reference_vstacks_sparse,
    gpu_compute_confusion_matrix,
    gpu_threshold
)
from cv2 import imread, imshow, imwrite, waitKey, ocl, UMat, blur
from numpy import array, uint8, uint16, sum as npsum, zeros
from server.db import CameraModel
from typing import no_type_check
from json import dump

ocl.setUseOpenCL(True)

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
    ref = get_reference_vstacks_sparse(camera=camera, ctag=ColourTag.RGB, indices=indices)
    imshow("Clouds", cloudimg)
    imshow("Cloud Masks", array(cmask*255, dtype=uint8))
    imshow("Reference", ref)
    waitKey(0)


@no_type_check
def thresholding():
    camera = Camera(CameraModel.DSLR)
    indices = array([0, 1, 2], dtype=uint16)
    gt_masks, _ = get_masks_vstacks_sparse(camera, indices)
    ref_imgs = get_reference_vstacks_sparse(camera, ColourTag.HSV, indices)

    print(f"GT masks shape: {gt_masks.shape}, dtype: {gt_masks.dtype}")
    print(f"Reference images shape: {ref_imgs.shape}, dtype: {ref_imgs.dtype}")

    analyzer = ROCAnalyzer(camera=camera)
    thresholded =gpu_threshold(ref_imgs, 1, [0, 85])

    imshow("GT Masks", array(gt_masks * 255, dtype=uint8))
    imshow("Thresholded", array(thresholded * 255, dtype=uint8))
    imshow("Reference Images", ref_imgs)
    waitKey(0)

    results = gpu_compute_confusion_matrix(gt_masks, thresholded)
    print(f"Confusion Matrix:\n{results}")


@no_type_check
def roc():
    camera = Camera(CameraModel.DSLR)
    analyzer = ROCAnalyzer(camera=camera, config=None)
    results = analyzer.analyze_roc(
        camera=camera,
        colortags=[ColourTag.HSV]
    )

    jsondict = {tagname: [] for tagname in results.keys()}
    
    for ctag, metrics in results.items():
        print(f"Color Tag: {ctag}")
        for metric in metrics:
            lower, upper = int(metric[0]), int(metric[1])
            tpr, fpr, precision, accuracy = metric[2], metric[3], metric[4], metric[5]

            jsondict[ctag].append({
                "lower": lower,
                "upper": upper,
                "tpr": tpr,
                "fpr": fpr,
                "precision": precision,
                "accuracy": accuracy
            })

    with open("roc_results.json", "w") as f:
        dump(jsondict, f, indent=4)

def verify_opencl_usage():
    """Verify OpenCL is working and being used"""
    print(f"OpenCL enabled: {ocl.useOpenCL()}")
    print(f"OpenCL available: {ocl.haveOpenCL()}")
    
    if ocl.haveOpenCL():
        # Get device info
        device = ocl.Device.getDefault()
        print(f"OpenCL device: {device.name()}")
        print(f"Device type: {device.type()}")
        print(f"Memory: {device.globalMemSize() // (1024*1024)} MB")
    
    # Test with a simple operation
    test_img = zeros((1000, 1000, 3), dtype=uint8)
    test_umat = UMat(test_img)
    result = blur(test_umat, (5, 5))
    print(f"UMat operations working: {isinstance(result, UMat)}")

if __name__ == "__main__":
    verify_opencl_usage()
    #roc()
    #dataset_vstack()
    #thresholding()