"""Plot rotation-matrix constraint drift under several update schemes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


def skew(omega: np.ndarray) -> np.ndarray:
    wx, wy, wz = omega
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation from a matrix using SVD."""
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def orthogonality_error(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))


def main() -> None:
    output = Path("public/linear_algebra_rotation_matrix_drift.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    omega = np.array([0.45, -0.25, 1.1])
    generator = skew(omega)
    dt = 0.05
    steps = 240
    times = np.arange(steps + 1) * dt

    euler = np.eye(3)
    qr_repaired = np.eye(3)
    svd_repaired = np.eye(3)
    exact = np.eye(3)
    increment = expm(dt * generator)

    euler_errors = [orthogonality_error(euler)]
    qr_errors = [orthogonality_error(qr_repaired)]
    svd_errors = [orthogonality_error(svd_repaired)]
    exact_errors = [orthogonality_error(exact)]
    euler_tips = [euler[:2, 0]]
    qr_tips = [qr_repaired[:2, 0]]
    svd_tips = [svd_repaired[:2, 0]]
    exact_tips = [exact[:2, 0]]

    for _ in range(steps):
        euler = euler + dt * generator @ euler
        qr_repaired, qr_scale = np.linalg.qr(qr_repaired + dt * generator @ qr_repaired)
        qr_signs = np.sign(np.diag(qr_scale))
        qr_signs[qr_signs == 0] = 1.0
        qr_repaired = qr_repaired @ np.diag(qr_signs)
        if np.linalg.det(qr_repaired) < 0:
            qr_repaired[:, -1] *= -1
        svd_repaired = project_to_rotation(svd_repaired + dt * generator @ svd_repaired)
        exact = increment @ exact

        euler_errors.append(orthogonality_error(euler))
        qr_errors.append(orthogonality_error(qr_repaired))
        svd_errors.append(orthogonality_error(svd_repaired))
        exact_errors.append(orthogonality_error(exact))
        euler_tips.append(euler[:2, 0])
        qr_tips.append(qr_repaired[:2, 0])
        svd_tips.append(svd_repaired[:2, 0])
        exact_tips.append(exact[:2, 0])

    euler_tips = np.asarray(euler_tips)
    qr_tips = np.asarray(qr_tips)
    svd_tips = np.asarray(svd_tips)
    exact_tips = np.asarray(exact_tips)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    axes[0].semilogy(times, np.maximum(euler_errors, 1e-16), color="#F44336", linewidth=2.2, label="Euler update")
    axes[0].semilogy(times, np.maximum(qr_errors, 1e-16), color="#FF9800", linewidth=2.0, label="Euler + QR")
    axes[0].semilogy(times, np.maximum(svd_errors, 1e-16), color="#9C27B0", linewidth=2.0, label="Euler + SVD")
    axes[0].semilogy(times, np.maximum(exact_errors, 1e-16), color="#2196F3", linewidth=2.0, label="Exponential update")
    axes[0].set_title("Orthogonality error over time")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel(r"$||R^T R-I||_F$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(euler_tips[:, 0], euler_tips[:, 1], color="#F44336", linewidth=2.0, label="Euler update")
    axes[1].plot(qr_tips[:, 0], qr_tips[:, 1], color="#FF9800", linewidth=2.0, label="Euler + QR")
    axes[1].plot(svd_tips[:, 0], svd_tips[:, 1], color="#9C27B0", linewidth=2.0, label="Euler + SVD")
    axes[1].plot(exact_tips[:, 0], exact_tips[:, 1], color="#2196F3", linewidth=2.0, label="Exponential update")
    circle = plt.Circle((0, 0), 1, fill=False, color="#607D8B", linestyle="--", linewidth=1.2, label="unit circle")
    axes[1].add_patch(circle)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("Tip of the first rotation axis")
    axes[1].set_xlabel("world x")
    axes[1].set_ylabel("world y")
    axes[1].set_xlim(-1.35, 1.35)
    axes[1].set_ylim(-1.35, 1.35)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left", frameon=True)

    fig.suptitle("Why rotation matrices need constraint maintenance", fontsize=14)
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
