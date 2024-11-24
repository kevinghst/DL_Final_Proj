import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def print_sample(sample):

    position_channel = sample[:, 0, :, :]  # [T, H, W] for position
    walls_channel = sample[:, 1, :, :]     # [T, H, W] for walls/doors
    position_channel = position_channel.cpu().numpy()
    walls_channel = walls_channel.cpu().numpy()

    steps, height, width = position_channel.shape
    for t in range(steps):
        print_image(position_channel[t], walls_channel[t])

    #plot_image(position_channel[1], walls_channel[1])
    #plot_trajectory(position_channel, walls_channel)
    plot_full_trajectory(position_channel, walls_channel)

def print_image(position, walls):

    height, width = position.shape
    print(f'height={height}, width={width}')
    for i in range(height):
        row = ""
        for j in range(width):
            if walls[i, j] > 0:
                row += "|"
            elif position[i, j] > 0:
                row += "X"
            else:
                row += "."
        print(row)

def plot_image(position, walls):

        fig = plt.figure(figsize=(12, 6))

        a = fig.add_subplot(1, 2, 1)
        im_a = a.imshow(position, cmap="viridis")
        a.set_title("Position - Timestep 0")
        fig.colorbar(im_a, ax=a)

        b = fig.add_subplot(1, 2, 2)
        im_b = b.imshow(walls, cmap="gray")
        b.set_title("Walls/Doors - Timestep 0")
        fig.colorbar(im_b, ax=b)

        fig.tight_layout()
        fig.savefig('traject.png')

def plot_trajectory(position_channel, walls_channel):

    steps = position_channel.shape[0]

    fig, axes = plt.subplots(steps, 2, figsize=(12, 6 * steps))

    if steps == 1:
        axes = axes[np.newaxis, :]

    for t in range(steps):
        ax_pos = axes[t, 0]
        im_pos = ax_pos.imshow(position_channel[t], cmap="viridis")
        ax_pos.set_title(f"Position - Timestep {t}")
        fig.colorbar(im_pos, ax=ax_pos)

        ax_wall = axes[t, 1]
        im_wall = ax_wall.imshow(walls_channel[t], cmap="gray")
        ax_wall.set_title(f"Walls/Doors - Timestep {t}")
        fig.colorbar(im_wall, ax=ax_wall)

    fig.tight_layout()
    fig.savefig('traject.png')

def plot_full_trajectory(position_channel, walls_channel):
    steps, height, width = position_channel.shape

    walls = walls_channel[0]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(walls, cmap="gray", alpha=0.6, label="Walls")  # Walls as background

    cmap = plt.cm.get_cmap("viridis", steps)  # Colormap with one color per step

    for t in range(steps):
        positions = np.argwhere(position_channel[t] > 0)  # Get positions of the agent
        for y, x in positions:
            ax.scatter(x, y, color=cmap(t), label=f"Timestep {t}" if t == 0 else "", s=20)

    ax.set_title("Agent Trajectory Over Static Walls")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=steps - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Timestep")
    fig.tight_layout()
    fig.savefig("traject.png")
    plt.close(fig)

