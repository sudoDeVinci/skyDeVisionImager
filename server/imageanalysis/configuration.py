from ..db import CameraModel

from enum import Enum
from os import listdir
from pathlib import Path
from functools import lru_cache
from typing import Final, Generic, TypeVar, Sequence, Optional


current_dir = Path(__file__).parent.resolve()


ModelType = TypeVar("ModelType", bound=CameraModel)


class Camera(Generic[ModelType]):
    """
    Camera configuration class that manages paths and operations for different camera models.
    """

    __slots__ = (
        "model",
        "image_folder",
        "blocked_images",
        "reference_images",
        "cloud_images",
        "sky_images",
        "cloud_masks",
        "sky_masks",
        "graphing_folder",
        "histograms",
        "pca",
        "roc",
        "cache",
        "calibration_folder",
        "camera_matrices",
        "training_images",
        "undistorted_images",
        "distorted_images",
        "calibration_configs",
    )

    def __init__(self, model: CameraModel) -> None:
        self.model: Final[CameraModel] = model
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Initialize all path attributes based on the camera model."""
        # Image Folders - Sky data related image paths
        self.image_folder: Path = current_dir.parent / "images" / self.model.value
        self.blocked_images: Path = self.image_folder / "blocked"
        self.reference_images: Path = self.image_folder / "reference"
        self.cloud_images: Path = self.image_folder / "cloud"
        self.sky_images: Path = self.image_folder / "sky"
        self.cloud_masks: Path = self.image_folder / "cloud_masks"
        self.sky_masks: Path = self.image_folder / "sky_masks"

        # graphing folders - Weather data graph related paths
        self.graphing_folder: Path = current_dir.parent / "graphs" / self.model.value
        self.histograms: Path = self.graphing_folder / "histograms"
        self.pca: Path = self.graphing_folder / "pca"
        self.roc: Path = self.graphing_folder / "roc"
        self.cache: Path = self.graphing_folder / "cache"

        # Calibration image folders - Calibration related image paths
        self.calibration_folder: Path = (
            current_dir.parent / "calibration" / self.model.value
        )
        self.camera_matrices: Path = self.calibration_folder / "camera_matrices"
        self.training_images: Path = self.calibration_folder / "training_images"
        self.undistorted_images: Path = self.calibration_folder / "undistorted_images"
        self.distorted_images: Path = self.calibration_folder / "distorted_images"
        self.calibration_configs: Path = self.calibration_folder / "calibration_configs"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Camera):
            return False
        return self.model == other.model

    def __hash__(self) -> int:
        return hash(self.model.value)

    def create_directories(self) -> None:
        """Create all directories for this camera instance."""
        for attr_name in dir(self):
            if not attr_name.startswith("_"):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, Path):
                    attr_value.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=1)
    def reference_images_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                self.reference_images / name
                for name in self.reference_images.iterdir()
                if name.is_file()
            )
        )

    @lru_cache(maxsize=1)
    def blocked_images_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                self.blocked_images / name
                for name in self.blocked_images.iterdir()
                if name.is_file()
            )
        )

    @lru_cache(maxsize=1)
    def cloud_images_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                self.cloud_images / name
                for name in self.cloud_images.iterdir()
                if name.is_file()
            )
        )

    @lru_cache(maxsize=1)
    def sky_images_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                self.sky_images / name
                for name in self.sky_images.iterdir()
                if name.is_file()
            )
        )

    @lru_cache(maxsize=1)
    def cloud_masks_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                self.cloud_masks / name
                for name in self.cloud_masks.iterdir()
                if name.is_file()
            )
        )

    @lru_cache(maxsize=1)
    def sky_masks_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                self.sky_masks / name
                for name in self.sky_masks.iterdir()
                if name.is_file()
            )
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def patchcheck(
        lista: Sequence[Path], listb: Sequence[Path]
    ) -> tuple[bool, Optional[str]]:
        """
        Check if two lists of paths have the same length and names.
        Returns a tuple of (bool, str) where bool indicates if they match,
        and str is an error message if they do not match.

        Args:
            lista(Sequence[Path]): First list of paths.
            listb(Sequence[Path]): Second list of paths.
        """
        if len(lista) != len(listb):
            return False, f"Length mismatch::{len(lista)} != {len(listb)}"

        for a, b in zip(lista, listb):
            if a.name != b.name:
                return False, f"Name mismatch::{a.name} != {b.name}"

        return True, None
