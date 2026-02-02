import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import proj3d
import colorsys


def fibonacci_sphere(n):
    """Uniformly distributed points on a sphere."""
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y, z))


def generate_colors(n, seed=42):
    """Generate a set of distinct colors with slight variation."""
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    colors = []
    for h in hues:
        s = 0.6 + rng.random() * 0.1
        v = 0.9 + rng.random() * 0.05
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))
    return colors


def spherical_tag_cloud_slider(tags, weights=None, radius=0.4, depth_scale=0.6):
    n = len(tags)
    pts_base = fibonacci_sphere(n)

    # Base font sizes (scaled by optional weights)
    if weights is None:
        base_sizes = np.full(n, 14.0)
    else:
        w = np.asarray(weights, float)
        w = (w - w.min()) / (w.max() - w.min() + 1e-9)
        base_sizes = 12 + 20 * w

    colors = generate_colors(n)

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95)

    # Create 2D text labels
    texts = []
    for i in range(n):
        texts.append(ax.text2D(
            0, 0, tags[i],
            ha="center", va="center",
            color=colors[i],
            fontsize=base_sizes[i]
        ))

    # Parameters for drawing the wireframe sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x_sphere_base = np.outer(np.cos(u), np.sin(v))
    y_sphere_base = np.outer(np.sin(u), np.sin(v))
    z_sphere_base = np.outer(np.ones_like(u), np.cos(v))
    sphere_plot = None  # Wireframe object placeholder

    # Radius slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Radius', 0.1, 1.2, valinit=radius)

    # Checkbox to toggle sphere visibility
    ax_checkbox = plt.axes([0.02, 0.8, 0.12, 0.12])
    ax_checkbox.set_frame_on(False)          # Remove frame border
    ax_checkbox.set_xticks([])
    ax_checkbox.set_yticks([])
    ax_checkbox.set_facecolor((0, 0, 0, 0))  # Transparent background

    check = CheckButtons(
        ax_checkbox,
        ['Show Sphere'],
        [False]        # Sphere hidden by default
    )

    lim = 0.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    show_sphere = [False]

    def update(val=None):
        nonlocal sphere_plot
        r = slider.val
        pts = pts_base * r

        # Compute camera direction vector
        cam_vec = np.array([
            np.cos(np.deg2rad(ax.elev)) * np.cos(np.deg2rad(ax.azim)),
            np.cos(np.deg2rad(ax.elev)) * np.sin(np.deg2rad(ax.azim)),
            np.sin(np.deg2rad(ax.elev))
        ])
        depth = pts @ cam_vec
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        sizes = base_sizes * (1 + depth_scale * (depth_norm - 0.5) * 2)
        sizes = np.clip(sizes, 8, 50)

        # Update text positions and sizes
        for i in range(n):
            x2, y2, _ = proj3d.proj_transform(
                pts[i, 0], pts[i, 1], pts[i, 2], ax.get_proj()
            )
            texts[i].set_position((x2, y2))
            texts[i].set_fontsize(sizes[i])

        # Add or remove the wireframe sphere
        if sphere_plot:
            sphere_plot.remove()
            sphere_plot = None
        if show_sphere[0]:
            sphere_plot = ax.plot_wireframe(
                x_sphere_base * r, y_sphere_base * r, z_sphere_base * r,
                color='gray', alpha=0.5, linewidth=0.7
            )

        fig.canvas.draw_idle()

    def toggle_sphere(label):
        show_sphere[0] = not show_sphere[0]
        update()

    slider.on_changed(update)
    check.on_clicked(toggle_sphere)

    update()

    def on_draw(event):
        update()

    fig.canvas.mpl_connect('motion_notify_event', on_draw)

    plt.show(block=True)


if __name__ == "__main__":
    tags = [
        "Argentina", "Australia", "Austria", "Brazil", "Canada", "China",
        "Denmark", "Egypt", "France", "Germany", "India", "Japan",
        "Mexico", "Netherlands", "Norway", "Russia", "Spain", "Sweden",
        "United Kingdom", "United States",
    ]

    spherical_tag_cloud_slider(tags)

