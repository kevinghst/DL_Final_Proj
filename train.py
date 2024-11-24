import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_low_energy_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for i in range(10):
            for batch in train_loader:
                break

        for batch in train_loader:
            states = batch.states.to(device, non_blocking=True)  # [B, T, Ch, H, W]
            actions = batch.actions.to(device, non_blocking=True)  # [B, T-1, 2]
            sample = states[60]
            break

            predictions = model(states, actions)  # [B, T, D]

            predicted_next_states = predictions[:, :-1, :]  # [B, T-1, D]
            predicted_last_state = predictions[:, -1, :]  # [B, D]

            # is this permitted?
            true_next_states = model.state_encoder(states[:, 1:, :, :, :].contiguous().view(-1, *states.shape[2:]))
            true_next_states = true_next_states.view_as(predicted_next_states)

            positive_energy = torch.norm(predicted_next_states - true_next_states, dim=2)
            shuffled_actions = actions[torch.randperm(actions.size(0))]
            shuffled_predictions = model(states, shuffled_actions)[:, :-1, :]
            negative_energy = torch.norm(shuffled_predictions - true_next_states, dim=2)

            contrastive_loss = F.relu(positive_energy - negative_energy + model.margin).mean()
            predictive_loss = mse_loss(predicted_next_states, true_next_states)
            loss = contrastive_loss + predictive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(sample)
        position_channel = sample[:, 0, :, :]  # [T, H, W] for position
        walls_channel = sample[:, 1, :, :]     # [T, H, W] for walls/doors
        position_channel = position_channel.cpu().numpy()
        walls_channel = walls_channel.cpu().numpy()

        def normalize(data):
            return (data - data.min()) / (data.max() - data.min()) if data.max() > 0 else data

        position_channel = normalize(position_channel)
        walls_channel = normalize(walls_channel)

        print(position_channel)
        print(walls_channel)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(position_channel[0], cmap="viridis")
        plt.title("Position - Timestep 0")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(walls_channel[0], cmap="gray")
        plt.title("Walls/Doors - Timestep 0")
        plt.colorbar()

        plt.tight_layout()
        plt.show()
        break
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


