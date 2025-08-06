from server.imageanalysis import (
    get_datasets_vstack,
    Camera,
    ColourTag,
    ROCAnalyzer,
    BoundaryRange,
    get_datasets_vstacks_sparse,
    get_masks_vstacks_sparse,
    get_reference_vstacks_sparse,
    compute_confusion_matrix
)
from cv2 import imread, imshow, imwrite, waitKey
from numpy import array, uint8, uint16, sum as npsum
from server.db import CameraModel
from typing import no_type_check
from json import dump


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

@no_type_check
def debug_roc():
    camera = Camera(CameraModel.DSLR)
    
    # Check data shapes first
    cloud_paths = list(camera.cloud_images_paths())
    sky_paths = list(camera.sky_images_paths())
    print(f"Cloud images: {len(cloud_paths)}")
    print(f"Sky images: {len(sky_paths)}")
    
    # Test mask loading
    indices = array([0, 1, 2])
    gt_masks, _ = get_masks_vstacks_sparse(camera, indices)
    ref_imgs = get_reference_vstacks_sparse(camera, ColourTag.RGB, indices)
    
    print(f"GT masks shape: {gt_masks.shape}, dtype: {gt_masks.dtype}")
    print(f"Reference images shape: {ref_imgs.shape}, dtype: {ref_imgs.dtype}")
    print(f"GT masks range: {gt_masks.min()} to {gt_masks.max()}")
    print(f"Ref images range: {ref_imgs.min()} to {ref_imgs.max()}")
    
    # Test thresholding
    analyzer = ROCAnalyzer(camera=camera)
    test_boundary = BoundaryRange(upper=uint8(200), lower=uint8(100))
    thresholded = analyzer.threshold(ref_imgs, 0, test_boundary)
    
    print(f"Thresholded shape: {thresholded.shape}, dtype: {thresholded.dtype}")
    print(f"Thresholded range: {thresholded.min()} to {thresholded.max()}")
    print(f"Thresholded sum: {npsum(thresholded)}")

@no_type_check
def test_small_roc():
    camera = Camera(CameraModel.DSLR)
    
    # Test with just a few images
    indices = array([0, 1, 2], dtype=uint16)
    gt_masks, _ = get_masks_vstacks_sparse(camera, indices)
    ref_imgs = get_reference_vstacks_sparse(camera, ColourTag.RGB, indices)
    
    # Test a single boundary
    analyzer = ROCAnalyzer(camera=camera)
    boundary = BoundaryRange(lower=uint8(100), upper=uint8(200))
    thresholded = analyzer.threshold(ref_imgs, 0, boundary)  # Test red channel

    imshow("GT Masks", array(gt_masks * 255, dtype=uint8))
    imshow("Thresholded", array(thresholded * 255, dtype=uint8))
    imshow("Reference Images", ref_imgs)
    waitKey(0)
    
    # Compute confusion matrix
    tpr, fpr, precision, accuracy = compute_confusion_matrix(gt_masks, thresholded)
    
    print(f"Ground truth positives: {npsum(gt_masks)}")
    print(f"Predicted positives: {npsum(thresholded)}")
    print(f"TPR: {tpr:.4f}, FPR: {fpr:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    roc()
    #dataset_vstack()
    #debug_roc()
    #test_small_roc()