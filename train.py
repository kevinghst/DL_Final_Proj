import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import WallDataset

def train_low_energy_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    all_states = train_loader.dataset.states
    all_actions = train_loader.dataset.actions

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        chunk_size = 10000 
        num_epochs_per_chunk = 2
        num_chunks = len(all_states) // chunk_size

        for chunk_id in range(num_chunks):
            chunk_states = all_states[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
            chunk_actions = all_actions[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]

            train_loader.dataset.states = chunk_states
            train_loader.dataset.actions = chunk_actions

            for epoch_chunk in range(num_epochs_per_chunk):
                for batch in train_loader:
                    states = batch.states.to(device, non_blocking=True)  # [B, T, Ch, H, W]
                    actions = batch.actions.to(device, non_blocking=True)  # [B, T-1, 2]

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

                print(f"Epoch chunk {epoch_chunk + 1}/{num_epochs_per_chunk}, Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


