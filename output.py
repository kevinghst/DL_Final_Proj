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
    ax.imshow(walls, cmap="gray", alpha=0.6, label="Walls") 
    for t in range(steps):
        im_pos = ax.imshow(
            position_channel[t],
            cmap="viridis",
            alpha=0.6,
            label=f"Timestep {t}",
        )
    ax.set_title("Agent Trajectory Over Static Walls")
    cbar = fig.colorbar(im_pos, ax=ax)
    cbar.set_label("Agent Intensity Over Time")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("traject.png")
    plt.close(fig)
