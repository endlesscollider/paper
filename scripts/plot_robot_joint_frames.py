"""Draw a two-link arm with joint axes and link frames attached.

Used by 系列/robot_foundations_deep_dive/01_关节自由度与三维旋转表示.md
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BG = "#FAFAFA"
LINK = "#90A4AE"
LINK2 = "#607D8B"
AXIS_Z = "#F44336"
AXIS_X = "#2196F3"
AXIS_Y = "#4CAF50"


def draw_frame(ax, origin, R, scale=0.28, label="", label_offset=(0.0, -0.10, 0.40),
               ha="center"):
    colors = [AXIS_X, AXIS_Y, AXIS_Z]
    names = ["x", "y", "z"]
    for i in range(3):
        v = R[:, i] * scale
        ax.quiver(*origin, *v, color=colors[i], linewidth=2.2, arrow_length_ratio=0.18)
        ax.text(*(origin + v * 1.18), names[i], color=colors[i], fontsize=9)
    if label:
        ax.text(*(origin + np.array(label_offset)), label, fontsize=11,
                fontweight="bold", ha=ha)


def main():
    fig = plt.figure(figsize=(9, 6.5), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)

    # Link 0: base. Link 1 along +z, link 2 bent.
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.9, 0.35, 1.45])

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=LINK, linewidth=10, solid_capstyle="round")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=LINK2, linewidth=8, solid_capstyle="round")

    # Joint 1 axis (vertical, through base)
    ax.quiver(-0.16, -0.30, 0.0, 0, 0, 0.72, color="#9C27B0", linewidth=3,
              arrow_length_ratio=0.16)
    ax.text(-0.30, -0.42, 0.76, r"joint 1 axis $z_1$", color="#9C27B0", fontsize=10)

    # Joint 2 axis: perpendicular to the arm plane, at p1
    ax.quiver(p1[0], p1[1] - 0.45, p1[2], 0, 0.78, 0, color="#FF9800",
              linewidth=3, arrow_length_ratio=0.14)
    ax.text(p1[0] + 0.10, p1[1] - 0.52, p1[2] - 0.24, r"joint 2 axis $z_2$",
            color="#FF9800", fontsize=10)

    # Frames at base, elbow, tip
    R_world = np.eye(3)
    R_elbow = np.array([[0.36, 0.0, -0.93],
                        [0.0, 1.0, 0.0],
                        [0.93, 0.0, 0.36]])
    R_tip = R_elbow
    draw_frame(ax, p0, R_world, scale=0.42, label="base frame {0}",
               label_offset=(0.30, 0.10, -0.28), ha="left")
    draw_frame(ax, p1, R_elbow, scale=0.30, label="elbow frame {1}",
               label_offset=(-0.05, -0.18, 0.62), ha="center")
    draw_frame(ax, p2, R_tip, scale=0.24, label="{2}",
               label_offset=(0.0, 0.0, 0.34), ha="center")

    ax.set_xlim(-0.4, 1.6)
    ax.set_ylim(-0.6, 1.0)
    ax.set_zlim(-0.2, 1.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Each link carries its own frame; each joint owns one axis",
                 fontsize=12)
    ax.view_init(elev=22, azim=-58)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    plt.tight_layout()
    fig.savefig("public/robot_joint_frames.png", dpi=160,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("wrote public/robot_joint_frames.png")


if __name__ == "__main__":
    main()
