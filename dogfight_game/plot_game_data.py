import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm


def plot_colourline(x, y, c, widths, ax):
    segments = []
    for i in np.arange(len(x) - 1):
        segments.append([[x[i], y[i]], [x[i + 1], y[i + 1]]])

    line_segments = LineCollection(
        segments, linewidths=widths, colors=c, linestyle="solid", zorder=1
    )
    ax.add_collection(line_segments)
    return


# @st.cache
def plotGameData(positions, hits, health, deathtimes=None):
    _, N_agents, steps = positions.shape
    f, ax = plt.subplots(figsize=(8, 8))

    ax.set_aspect("equal", "box")
    ax.set_xlim([-13, 13])
    ax.set_ylim([-13, 13])

    # colors = np.random.rand(N_agents, 3) * 0.6 + 0.4
    colors = [plt.cm.tab20(i % 20) for i in range(N_agents - 1)]
    colors.insert(0, (0.0, 0.0, 0.0, 1.0))

    ax.scatter(positions[0, :, 0], positions[1, :, 0], c=colors)

    hit_segs = []
    hit_seg_colors = []

    for i in range(N_agents):
        plot_colourline(
            positions[0, i, :],
            positions[1, i, :],
            # np.hstack([np.tile(colors[i, :], (steps, 1)), health[i:i+1, :].T**2]),
            colors[i],
            (2 * health[i, :] + 0.5) * health[i, :] ** (0.5),
            ax,
        )

        for t in range(steps):
            for j in range(N_agents):
                if hits[i, j, t]:
                    hit_segs.append(
                        [
                            [positions[0, i, t], positions[1, i, t]],
                            [positions[0, j, t], positions[1, j, t]],
                        ]
                    )
                    hit_seg_colors.append(colors[i])

    ax.add_collection(
        LineCollection(
            hit_segs, colors=hit_seg_colors, linewidths=0.5, alpha=0.3, zorder=0
        )
    )

    if deathtimes is not None and not np.all(np.isnan(deathtimes)):
        deathXs = []
        deathYs = []
        deathColors = []
        for i, t in enumerate(deathtimes):
            if np.isnan(t):
                continue
            t = int(t)
            deathXs.append(positions[0, i, t])
            deathYs.append(positions[1, i, t])
            deathColors.append(colors[i])
        ax.scatter(deathXs, deathYs, c=deathColors, s=80, marker="x", zorder=2)

    return f
