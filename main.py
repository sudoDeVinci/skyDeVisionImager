from server.imageanalysis import (
    Camera,
    ColourTag,
    ROCAnalyzer,
    AnalysisConfiguration,
    get_datasets_vstacks_sparse,
    get_masks_vstacks_sparse,
    get_reference_vstacks_sparse,
    verify_gpu_setup,
)
from cv2 import imshow, waitKey, ocl
from numpy import array, uint8, zeros, uint16, float32
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
    config = AnalysisConfiguration(
        strata_count=uint16(20),
        strata_size=uint16(30),
        boundary_width=uint8(20),
        jaccard_threshold=float32(0.25),
        max_workers=uint8(4),
        caching=False,
        gpu_caching=True
    )
    analyzer = ROCAnalyzer(camera=camera, config=config)
    results = analyzer.analyze_roc(
        camera=camera,
        colortags=[ColourTag.HSV, ColourTag.RGB, ColourTag.YBR],
    )

    jsondict = {
        "config": config.to_dict(),
        "results": {}
    }
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

    with open("roc_results.json", "w") as f:
        dump(jsondict, f, indent=4)


def verify_opencl_usage():
    """Verify OpenCL is working and being used"""
    return verify_gpu_setup()


def load_batch():
    camera = Camera(CameraModel.DSLR)
    samples = array([[0, 1, 2, 3]], dtype=uint8)
    analyzer = ROCAnalyzer(camera=camera, config=None)

    gtmask, refimg = analyzer._load_batch(
        camera=camera, tag=ColourTag.RGB, bootstrap_samples=samples
    )

    imshow("Ground Truth Mask", gtmask)
    imshow("Reference Image", refimg)
    waitKey(0)


def load_batches():
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


def subprocess_nvidia_test():
    """Function to run in subprocess - must be at module level for pickling"""
    import os

    # Check if environment variable is inherited
    pyopencl_ctx = os.environ.get("PYOPENCL_CTX", "Not set")
    print(f"Subprocess PYOPENCL_CTX: {pyopencl_ctx}")

    # Try to select NVIDIA device in subprocess
    nvidia_selected = select_nvidia_device()
    print(f"Subprocess NVIDIA selection: {nvidia_selected}")

    # Test OpenCV with NVIDIA
    opencv_result = test_opencv_with_nvidia()
    print(f"Subprocess OpenCV test: {opencv_result}")

    return nvidia_selected and opencv_result


def test_nvidia_selection_in_subprocess():
    """Test if NVIDIA device selection works in subprocess"""
    from concurrent.futures import ProcessPoolExecutor

    print("\n=== Testing NVIDIA Selection in Subprocess ===")

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            subprocess_nvidia_test
        )  # Now using module-level function
        try:
            result = future.result(timeout=30)
            print(f"Subprocess NVIDIA test result: {result}")
            return result
        except Exception as e:
            print(f"Subprocess NVIDIA test failed: {e}")
            return False


def list_opencl_devices():
    """List all available OpenCL devices using pyopencl"""
    try:
        import pyopencl as cl

        print("=== Available OpenCL Platforms and Devices ===")
        platforms = cl.get_platforms()

        for p_idx, platform in enumerate(platforms):
            print(f"\nPlatform {p_idx}: {platform.name}")
            print(f"  Vendor: {platform.vendor}")
            print(f"  Version: {platform.version}")

            devices = platform.get_devices()
            for d_idx, device in enumerate(devices):
                print(f"  Device {d_idx}: {device.name}")
                print(f"    Type: {cl.device_type.to_string(device.type)}")
                print(f"    Global Memory: {device.global_mem_size // (1024*1024)} MB")
                print(f"    Local Memory: {device.local_mem_size // 1024} KB")
                print(f"    Max Compute Units: {device.max_compute_units}")
                print(f"    Max Work Group Size: {device.max_work_group_size}")

                # Check if this is NVIDIA
                if "NVIDIA" in device.name or "GeForce" in device.name:
                    print(f"    *** NVIDIA GPU FOUND ***")

    except ImportError:
        print("PyOpenCL not installed. Install with: pip install pyopencl")
    except Exception as e:
        print(f"Error listing OpenCL devices: {e}")


def select_nvidia_device():
    """Set environment variable to force OpenCL to use NVIDIA device"""
    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        nvidia_device_found = False

        for p_idx, platform in enumerate(platforms):
            devices = platform.get_devices()
            for d_idx, device in enumerate(devices):
                if "NVIDIA" in device.name or "GeForce" in device.name:
                    print(f"Found NVIDIA device: {device.name}")
                    print(f"Platform: {p_idx}, Device: {d_idx}")

                    # Set environment variable for OpenCL device selection
                    import os

                    device_string = f"{p_idx}:{d_idx}"
                    os.environ["PYOPENCL_CTX"] = device_string
                    print(f"Set PYOPENCL_CTX={device_string}")

                    nvidia_device_found = True
                    break
            if nvidia_device_found:
                break

        if not nvidia_device_found:
            print("No NVIDIA device found!")
            return False

        return True

    except ImportError:
        print("PyOpenCL not installed. Install with: pip install pyopencl")
        return False
    except Exception as e:
        print(f"Error selecting NVIDIA device: {e}")
        return False


def test_opencv_with_nvidia():
    """Test OpenCV UMat creation after selecting NVIDIA device"""
    from cv2 import UMat, ocl
    from numpy import zeros, uint8

    print("=== Testing OpenCV with NVIDIA Device ===")

    # Force OpenCL reinitialization
    ocl.setUseOpenCL(False)
    ocl.setUseOpenCL(True)

    print(f"OpenCL available: {ocl.haveOpenCL()}")
    print(f"OpenCL enabled: {ocl.useOpenCL()}")

    try:
        # Try larger allocation that would fail on iGPU
        arr = zeros((5000, 5000), dtype=uint8)  # ~25MB
        umat = UMat(arr)
        print(
            f"✓ Successfully created large UMat: {arr.size} elements ({arr.nbytes//1024//1024} MB)"
        )
        return True
    except Exception as e:
        print(f"✗ Failed to create UMat: {e}")
        return False


def test_opencl_in_threadpool():
    """Test OpenCL functionality with ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def thread_opencv_test(thread_id):
        """Test OpenCV UMat creation in a thread"""
        from cv2 import UMat, ocl
        from numpy import zeros, uint8

        print(f"Thread {thread_id}: OpenCL available: {ocl.haveOpenCL()}")
        print(f"Thread {thread_id}: OpenCL enabled: {ocl.useOpenCL()}")

        try:
            # Try creating UMat in thread
            arr = zeros((2000, 2000), dtype=uint8)  # ~4MB
            umat = UMat(arr)
            print(
                f"✓ Thread {thread_id}: Successfully created UMat: {arr.size} elements ({arr.nbytes//1024//1024} MB)"
            )
            return True
        except Exception as e:
            print(f"✗ Thread {thread_id}: Failed to create UMat: {e}")
            return False

    print("\n=== Testing OpenCL in ThreadPoolExecutor ===")

    # Test with multiple threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(thread_opencv_test, i) for i in range(3)]

        results = []
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result(timeout=10)
                results.append(result)
                print(f"Thread result {i+1}: {result}")
            except Exception as e:
                print(f"Thread {i+1} failed: {e}")
                results.append(False)

    all_passed = all(results)
    print(f"\nAll threads passed: {all_passed}")
    return all_passed


def test_actual_roc_scenario():
    """Test the actual ROC scenario with ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from cv2 import UMat, repeat
    from numpy import zeros, uint8

    def simulate_roc_work(tag_name):
        """Simulate the work done in _analyze_channel_roc"""
        print(f"Starting ROC work for {tag_name}")

        try:
            # Simulate loading masks (similar to your actual code)
            ground_truth = zeros((30000, 400), dtype=uint8)
            boundary_masks = zeros((300000, 400), dtype=uint8)  # 10 boundaries

            # Convert to UMat (this is where your code was failing)
            gt_umat = UMat(ground_truth)
            boundary_umat = UMat(boundary_masks)

            print(f"✓ {tag_name}: Created UMats successfully")

            # Simulate the cv2.repeat operation that was causing issues
            repeated_gt = repeat(gt_umat, 10, 1)
            print(f"✓ {tag_name}: cv2.repeat successful")

            return True

        except Exception as e:
            print(f"✗ {tag_name}: Failed - {e}")
            return False

    print("\n=== Testing Actual ROC Scenario with Threads ===")

    color_tags = ["HSV", "RGB", "YCrCb"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(simulate_roc_work, tag) for tag in color_tags]

        results = {}
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                # We can't easily map back to tag name, so just track success
                print(f"ROC simulation result: {result}")
            except Exception as e:
                print(f"ROC simulation failed: {e}")

    print("ROC simulation complete")


if __name__ == "__main__":
    from cv2 import UMat, ocl, bitwise_not, bitwise_and
    from time import sleep

    # First, list all available devices
    list_opencl_devices()

    # Try to select NVIDIA device
    if select_nvidia_device():
        print("Successfully selected NVIDIA device")
    else:
        print("Failed to select NVIDIA device, using default")

    if verify_opencl_usage():
        print("GPU setup verified successfully!")
    else:
        print("Warning: GPU setup issues detected")

    # If thread tests pass, we can try the actual ROC analysis
    roc()
