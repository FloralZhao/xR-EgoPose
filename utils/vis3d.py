import matplotlib
matplotlib.use('Agg')
import numpy as np

CM_TO_M = 100

def show3Dpose(channels, ax, gt, mm=True):
    vals = channels.reshape((16, 3))
    if mm:
        channels *= CM_TO_M
    if gt:
        color = "#3498db"
    else:
        color = "#e74c3c"
    I = np.array([0, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 2, 5, 2])  # start points
    J = np.array([1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 8, 12, 5])  # end points

    for i in range(16):
        ax.scatter(vals[i][0], vals[i][1], vals[i][2], c=color)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=color)

    RADIUS = 50  # space around the subject
    xroot, yroot, zroot = vals[8, 0], vals[8, 1], vals[8, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([RADIUS + zroot, -RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
