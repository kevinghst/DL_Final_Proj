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

    plot_image(position_channel[1], walls_channel[1])

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

