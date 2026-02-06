import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import proj3d
import colorsys

# ---------- geometry ----------
def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y, z))

def generate_colors(n, seed=42):
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

# ---------- main ----------
class TagCloud3D:
    def __init__(self, tags, radius=0.4, depth_scale=0.6):
        self.tags = tags
        self.radius = radius
        self.depth_scale = depth_scale
        self.n = len(tags)
        self.pts_base = fibonacci_sphere(self.n)
        self.colors = generate_colors(self.n)

        # Figure
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95)

        # Wireframe sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        self.x_sphere = np.outer(np.cos(u), np.sin(v))
        self.y_sphere = np.outer(np.sin(u), np.sin(v))
        self.z_sphere = np.outer(np.ones_like(u), np.cos(v))
        self.sphere_visible = True
        self.sphere_plot = self.ax.plot_wireframe(
            self.x_sphere * self.radius,
            self.y_sphere * self.radius,
            self.z_sphere * self.radius,
            color='gray', alpha=0.5, linewidth=0.7
        )

        # Base font sizes
        self.base_sizes = np.full(self.n, 14.0)

        # Create text objects once
        self.texts = []
        for i, tag in enumerate(tags):
            t = self.ax.text2D(0, 0, tag, ha='center', va='center', color=self.colors[i], fontsize=self.base_sizes[i])
            self.texts.append(t)

        # Slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Radius', 0.1, 1.2, valinit=self.radius)
        self.slider.on_changed(self.update)

        # CheckButtons
        ax_checkbox = plt.axes([0.02, 0.8, 0.12, 0.12])
        ax_checkbox.set_frame_on(False)
        ax_checkbox.set_xticks([])
        ax_checkbox.set_yticks([])
        ax_checkbox.set_facecolor((0, 0, 0, 0))
        from matplotlib.widgets import CheckButtons
        self.check = CheckButtons(ax_checkbox, ['Show Sphere'], [self.sphere_visible])
        self.check.on_clicked(self.toggle_sphere)

        # Axis settings
        lim = 0.8
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_axis_off()

        # Connect rotation
        self.fig.canvas.mpl_connect('motion_notify_event', lambda event: self.update())

        self.update()
        plt.show(block=True)

    def toggle_sphere(self, label):
        self.sphere_visible = not self.sphere_visible
        if self.sphere_visible:
            self.sphere_plot = self.ax.plot_wireframe(
                self.x_sphere * self.radius,
                self.y_sphere * self.radius,
                self.z_sphere * self.radius,
                color='gray', alpha=0.5, linewidth=0.7
            )
        else:
            if self.sphere_plot:
                self.sphere_plot.remove()
                self.sphere_plot = None
        self.fig.canvas.draw_idle()

    def update(self, val=None):
        r = self.slider.val
        pts = self.pts_base * r

        # Compute camera direction
        cam_vec = np.array([
            np.cos(np.deg2rad(self.ax.elev)) * np.cos(np.deg2rad(self.ax.azim)),
            np.cos(np.deg2rad(self.ax.elev)) * np.sin(np.deg2rad(self.ax.azim)),
            np.sin(np.deg2rad(self.ax.elev))
        ])
        depth = pts @ cam_vec
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        sizes = self.base_sizes * (1 + self.depth_scale * (depth_norm - 0.5) * 2)
        sizes = np.clip(sizes, 8, 50)

        # Update text positions and sizes
        for i in range(self.n):
            x2, y2, _ = proj3d.proj_transform(pts[i,0], pts[i,1], pts[i,2], self.ax.get_proj())
            self.texts[i].set_position((x2, y2))
            self.texts[i].set_fontsize(sizes[i])

        # Update sphere if visible
        if self.sphere_visible and self.sphere_plot:
            self.sphere_plot.remove()
            self.sphere_plot = self.ax.plot_wireframe(
                self.x_sphere * r,
                self.y_sphere * r,
                self.z_sphere * r,
                color='gray', alpha=0.5, linewidth=0.7
            )

        self.fig.canvas.draw_idle()


# ---------- run ----------
if __name__ == "__main__":
    tags = [
        "Argentina", "Australia", "Austria", "Brazil", "Canada", "China",
        "Denmark", "Egypt", "France", "Germany", "India", "Japan",
        "Mexico", "Netherlands", "Norway", "Russia", "Spain", "Sweden",
        "United Kingdom", "United States",
    ]

    TagCloud3D(tags)