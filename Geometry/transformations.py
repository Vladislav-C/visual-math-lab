import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# Basic shape
triangle = np.array([
    [1.0, 1.0],
    [5.0, 7.0],
    [8., 7.],
    [1.0, 1.0]
])


### Canvas setup ###
fig, ax = plt.subplots(figsize=(9, 7))
ax.legend(loc="lower right")

# Make axes cross at (0, 0)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_aspect('equal', adjustable='box')

ax.set_xlim(-10, 15)
ax.set_ylim(-10, 15)
ax.grid(True)

# Show original shape (static)
ax.plot(triangle[:, 0], triangle[:, 1], 'o-', color = "cornflowerblue", label="Original")


# ---------- choose transformation type ----------
mode = "translate"

frames = []
n_frames = 60

for step in range(n_frames):
    alpha = step / (n_frames - 1)   # goes 0 → 1

    if mode == "translate":
        A = np.eye(2)
        t = np.array([2.0 * alpha, 1.0 * alpha])

    elif mode == "rotate":
        theta = np.deg2rad(45 * alpha)
        A = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        t = np.array([0.0, 0.0])

    elif mode == "reflect":
        dif = -1.0 - 1. # end - start
        A = np.array([
            [1.0 + dif * alpha, 0.0],
            [0.0, 1.0]
        ])
        t = np.array([0.0, 0.0])

    elif mode == "scale":
        scale = 1 + 0.8  * alpha
        A = np.array([
            [scale, 0.0],
            [0.0, scale]
        ])
        t = np.array([0.0, 0.0])
    
    else:
        # default no change
        A = np.eye(2)
        t = np.array([0.0, 0.0])

    # apply transformation: x' = A x + t
    tri_new = triangle @ A.T + t

    line, = ax.plot(tri_new[:, 0], tri_new[:, 1], 'o--', color = "darkorange")
    frames.append([line])


anim = ArtistAnimation(fig, frames, interval=50, blit=True, repeat=True)
plt.show()