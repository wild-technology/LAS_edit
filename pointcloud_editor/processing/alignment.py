"""ICP registration and manual alignment utilities."""
import numpy as np

from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


def icp_align(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    initial_transform: np.ndarray | None = None,
    max_distance: float = 1.0,
    max_iterations: int = 50,
) -> np.ndarray:
    """Run ICP registration to align source to target.

    Args:
        source_xyz: float32 (N, 3) source points
        target_xyz: float32 (M, 3) target points
        initial_transform: float64 (4, 4) initial guess (identity if None)
        max_distance: maximum correspondence distance
        max_iterations: max ICP iterations

    Returns:
        float64 (4, 4) refined transform
    """
    try:
        import open3d as o3d

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_xyz.astype(np.float64))

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_xyz.astype(np.float64))

        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max_distance * 2, max_nn=30
            )
        )

        init = initial_transform if initial_transform is not None else np.eye(4)

        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_distance,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            ),
        )

        logger.info(
            f"ICP fitness={reg.fitness:.4f}, RMSE={reg.inlier_rmse:.6f}"
        )
        return np.array(reg.transformation, dtype=np.float64)

    except ImportError:
        logger.error("Open3D required for ICP alignment")
        return initial_transform if initial_transform is not None else np.eye(4)


def build_translation_matrix(dx: float, dy: float, dz: float) -> np.ndarray:
    """Build a 4x4 translation matrix."""
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = dx
    m[1, 3] = dy
    m[2, 3] = dz
    return m


def build_rotation_matrix(
    angle_x: float, angle_y: float, angle_z: float, pivot: np.ndarray | None = None
) -> np.ndarray:
    """Build a 4x4 rotation matrix (Euler angles in degrees) around a pivot point."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("xyz", [angle_x, angle_y, angle_z], degrees=True)
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = rot.as_matrix()

    if pivot is not None:
        pivot = np.asarray(pivot, dtype=np.float64)
        T_to = build_translation_matrix(-pivot[0], -pivot[1], -pivot[2])
        T_back = build_translation_matrix(pivot[0], pivot[1], pivot[2])
        return T_back @ R @ T_to

    return R


def decompose_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a 4x4 affine matrix into translation and Euler angles (degrees).

    Returns:
        (translation (3,), euler_angles (3,) in degrees)
    """
    from scipy.spatial.transform import Rotation

    translation = transform[:3, 3].copy()
    rotation = Rotation.from_matrix(transform[:3, :3])
    euler = rotation.as_euler("xyz", degrees=True)
    return translation, euler
