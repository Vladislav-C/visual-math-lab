import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Basic shape
shape = np.array([
    [1.0, 1.0],
    [5.0, 7.0],
    [8., 7.],
    [1.0, 1.0]
])


def performTrans(shape,
                 mode="translate",
                 translation=(0.0, 0.0),
                 angle_deg=0.0,
                 scale=1,
                 axis="x",
                 animation=True,
                 num_frames=60,
                 repeat=False, 
                 title="Examples of Geometric Transformations in 2D"
                 ):

    # Make sure it's a NumPy array with floating values
    shape = np.asarray(shape, dtype=float)


    ### Canvas setup ###
    fig, ax = plt.subplots(figsize=(9, 7))

    # Make axes cross at (0, 0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(title)

    # Original shape
    ax.plot(shape[:, 0], shape[:, 1], 'o-', color = "cornflowerblue", label="Original")


    ### Transformation ###
    mode = mode.lower()
    axis = axis.lower()

    def get_A_t(alpha):
        """Return tranformation matrix A and translation t for given progress alpha [0, 1]"""
        if mode == "translate":
            dx, dy = translation
            A = np.eye(2)
            t = alpha * np.array([dx, dy], dtype=float)

        elif mode == "rotate":
            theta = np.deg2rad(angle_deg * alpha)
            A = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            t = np.zeros(2)

        elif mode == "reflect":
            if axis == "y":
                A = np.array([
                    [1.0 + (-2) * alpha, 0.0],
                    [0.0, 1.0]
                ])
            else:
                A = np.array([
                    [1.0, 0.0],
                    [0.0, 1.0 + (-2) * alpha]
                ])
            t = np.zeros(2)

        elif mode == "scale":
            s_val= 1 + (scale - 1)  * alpha
            A = np.array([
                [s_val, 0.0],
                [0.0, s_val]
            ])
            t = np.zeros(2)
        
        else:
            # default no change
            A = np.eye(2)
            t = np.zeros(2)

        return A, t

    # Choose axis limits based on original and final positions
    A_final, t_final = get_A_t(1.0)
    shape_final = shape @ A_final.T + t_final
    all_points = np.vstack([shape, shape_final]) # --> (2N, 2)

    x_min, y_min = all_points.min(axis=0)   # axis = 0 --> go column by column
    x_max, y_max = all_points.max(axis=0)

    # Small padding
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    pad_x = 0.2 * dx
    pad_y = 0.2 * dy

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    if not animation:
        A, t = get_A_t(1.0)
        shp_new = shape @ A.T + t # apply transformation: x' = A x + t
        ax.plot(shp_new[:, 0], shp_new[:, 1], 'o--', color = "darkorange")
        ax.legend(loc="lower right")
        plt.show()
        return

    def update(frame):
        alpha = frame / (num_frames - 1)
        A, t = get_A_t(alpha)
        shp_new = shape @ A.T + t
        line = ax.plot(shp_new[:, 0], shp_new[:, 1], 'o--', color = "darkorange")
        return line


    anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True, repeat=repeat)
    ax.legend(loc="lower right")
    plt.show()
    return anim


anim = performTrans(shape, mode="scale", scale=5)
#anim.save("scale.gif", writer=PillowWriter(fps=30))