import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# ---------- Geometry ----------
def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y, z))

# ---------- OpenGL Widget ----------
class GLTagCloud(QOpenGLWidget):
    def __init__(self, tags):
        super().__init__()
        self.tags = tags
        self.n = len(tags)
        self.radius = 1.0
        self.angle_x = 0
        self.angle_y = 0
        self.last_pos = None
        self.base_pts = fibonacci_sphere(self.n)
        # Яркие базовые цвета для каждой надписи
        self.colors = 0.5 + 0.5 * np.random.rand(self.n, 3)
        self.show_sphere = True

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0, 0, 0, 1)
        glutInit()

        # --- Освещение для сферы ---
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h != 0 else 1, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        cam_pos = np.array([0, 0, 5])
        gluLookAt(*cam_pos, 0, 0, 0, 0, 1, 0)

        # --- Вращение сферы ---
        glRotatef(self.angle_x, 1, 0, 0)
        glRotatef(self.angle_y, 0, 1, 0)

        # --- Сфера ---
        if self.show_sphere:
            glColor3f(0.5, 0.5, 0.5)
            glLineWidth(3.0)
            glutWireSphere(self.radius, 30, 30)
            glLineWidth(2.0)

        # --- Предварительный расчёт матрицы поворота для яркости ---
        angle_x_rad = np.radians(self.angle_x)
        angle_y_rad = np.radians(self.angle_y)
        cx, sx = np.cos(angle_x_rad), np.sin(angle_x_rad)
        cy, sy = np.cos(angle_y_rad), np.sin(angle_y_rad)

        # Камера смотрит вдоль -Z, в eye-space направление к камере можно взять как (0,0,1). [web:36][web:44]
        dir_to_cam = np.array([0.0, 0.0, 1.0])

        # --- Теги ---
        for i, tag in enumerate(self.tags):
            base = self.base_pts[i] * self.radius

            # Поворачиваем точку так же, как сцену (Y, затем X),
            # чтобы получить её положение в координатах камеры.
            x, y, z = base

            # Поворот вокруг оси Y
            x2 =  cy * x + sy * z
            y2 =  y
            z2 = -sy * x + cy * z

            # Поворот вокруг оси X
            x3 = x2
            y3 =  cx * y2 - sx * z2
            z3 =  sx * y2 + cx * z2

            # Нормаль = радиус-вектор точки
            normal = np.array([x3, y3, z3])
            normal /= np.linalg.norm(normal)

            # Яркость: чем ближе к камере (больше z3), тем ярче
            brightness = np.clip(np.dot(normal, dir_to_cam), 0.2, 1.0)

            r, g, b = self.colors[i]
            color = np.array([r, g, b]) * brightness

            # Позиция точки в мире (как раньше)
            pt = base

            # Масштаб текста по расстоянию до камеры
            world_pt = pt
            dir_world = cam_pos - world_pt
            dist = np.linalg.norm(dir_world)
            scale = np.clip(0.002 * 5 / dist, 0.001, 0.005)

            glPushMatrix()
            glTranslatef(*pt)
            # Разворачиваем текст лицом к камере
            glRotatef(-self.angle_y, 0, 1, 0)
            glRotatef(-self.angle_x, 1, 0, 0)
            glScalef(scale, scale, scale)

            # Центрирование текста по ширине
            text_width = sum([glutStrokeWidth(GLUT_STROKE_ROMAN, ord(c)) for c in tag])
            glTranslatef(-text_width / 2.0, 0, 0)

            # --- динамическая толщина линий по яркости ---
            min_w, max_w = 1.0, 6.0
            line_width = min_w + (max_w - min_w) * (brightness - 0.2) / (1.0 - 0.2)
            line_width = float(np.clip(line_width, min_w, max_w))  # на всякий случай [web:45][web:55]

            glDisable(GL_LIGHTING)
            glLineWidth(line_width)
            glColor3f(*color)
            for c in tag:
                glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(c))
            glLineWidth(1.0)  # сброс
            glEnable(GL_LIGHTING)

            glPopMatrix()

    # ---------- Mouse Rotation ----------
    def mousePressEvent(self, event):
        self.last_pos = (event.position().x(), event.position().y())

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        x, y = event.position().x(), event.position().y()
        dx = x - self.last_pos[0]
        dy = y - self.last_pos[1]
        self.angle_x += dy * 0.5
        self.angle_y += dx * 0.5
        self.last_pos = (x, y)

# ---------- GUI ----------
class TagCloudWindow(QtWidgets.QWidget):
    def __init__(self, tags):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.gl_widget = GLTagCloud(tags)
        layout.addWidget(self.gl_widget, 1)

        # Slider радиуса
        slider_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(slider_layout)
        slider_layout.addWidget(QtWidgets.QLabel("Radius"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(200)
        self.slider.setValue(int(self.gl_widget.radius * 50))
        self.slider.valueChanged.connect(self.on_radius_change)
        slider_layout.addWidget(self.slider)

        # Checkbox видимости сферы
        self.checkbox = QtWidgets.QCheckBox("Show Sphere")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.on_checkbox_change)
        layout.addWidget(self.checkbox)

    def on_radius_change(self, value):
        self.gl_widget.radius = value / 50

    def on_checkbox_change(self, state):
        self.gl_widget.show_sphere = bool(state)

# ---------- Run ----------
if __name__ == "__main__":
    glutInit()
    app = QtWidgets.QApplication(sys.argv)
    tags = [
        "Argentina", "Australia", "Austria", "Brazil", "Canada", "China",
        "Denmark", "Egypt", "France", "Germany", "India", "Japan",
        "Mexico", "Netherlands", "Norway", "Russia", "Spain", "Sweden",
        "United Kingdom", "United States",
    ]
    w = TagCloudWindow(tags)
    w.resize(1000, 1000)
    w.show()
    sys.exit(app.exec())
