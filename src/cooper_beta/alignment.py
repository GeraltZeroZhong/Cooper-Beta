import numpy as np

class PCAAligner:
    """
    Use PCA to align the principal axis of a point cloud to the Z axis.
    """
    def __init__(self):
        self.center = None
        self.rotation_matrix = None
        self.eigenvalues = None

    def fit(self, points):
        """
        Compute the principal axes of the point cloud.

        For best results, pass only C-alpha coordinates from beta-sheet residues.

        Args:
            points (np.ndarray): Coordinate array with shape ``(N, 3)``.
        """
        coords = np.array(points)
        if coords.shape[0] < 3:
            raise ValueError("At least three points are required for PCA alignment.")

        # 1. Compute the centroid and center the coordinates.
        self.center = np.mean(coords, axis=0)
        centered_coords = coords - self.center

        # 2. Build the covariance matrix.
        # rowvar=False means columns are variables (x, y, z) and rows are samples.
        cov_matrix = np.cov(centered_coords, rowvar=False)

        # 3. Eigen decomposition.
        # eigh is appropriate for symmetric matrices such as covariance matrices,
        # and is usually more stable than eig here.
        # eig_vals are returned in ascending order (min -> max), and each column
        # of eig_vecs is a corresponding eigenvector.
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        self.eigenvalues = eig_vals
        self.rotation_matrix = eig_vecs
        
        # Debug helper: variance contribution of each PCA axis.
        # variance_ratio = eig_vals / np.sum(eig_vals)
        # print(f"PCA Variance Ratios (X, Y, Z): {variance_ratio}")

    def transform(self, points):
        """
        Project the point cloud into the PCA coordinate system.

        After transformation, the Z axis corresponds to the dominant principal
        axis of the original point cloud.
        """
        if self.rotation_matrix is None:
            raise RuntimeError("Aligner is not fitted. Call fit() first.")

        coords = np.array(points)

        # 1. Translate to the centered coordinate system.
        centered = coords - self.center

        # 2. Rotate by projecting onto the eigenvector basis.
        # result = centered @ eigenvectors
        # Column 0 corresponds to the smallest eigenvalue direction (new X),
        # column 1 to the middle eigenvalue direction (new Y), and column 2 to
        # the largest eigenvalue direction (new Z, the principal axis).
        transformed = np.dot(centered, self.rotation_matrix)

        return transformed

    def fit_transform(self, points):
        """Convenience wrapper that runs ``fit`` followed by ``transform``."""
        self.fit(points)
        return self.transform(points)

# --- Simple self-check ---
if __name__ == "__main__":
    # Create an elongated cylinder-like point cloud to mimic a beta barrel.
    t = np.linspace(0, 20, 100)
    x = np.cos(t) + np.random.normal(0, 0.1, 100)
    y = np.sin(t) + np.random.normal(0, 0.1, 100)
    z = t * 2

    original_points = np.column_stack([x, y, z])

    aligner = PCAAligner()
    transformed_points = aligner.fit_transform(original_points)

    print("Original Z-axis variance:", np.var(original_points[:, 2]))
    print("Transformed Z-axis variance:", np.var(transformed_points[:, 2]))
    print("Transformed X-axis variance:", np.var(transformed_points[:, 0]))

    if np.var(transformed_points[:, 2]) > np.var(transformed_points[:, 0]):
        print("Alignment test passed: the principal axis is aligned to Z.")
    else:
        print("Alignment test failed: Z is not the axis with the largest variance.")
