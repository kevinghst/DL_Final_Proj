import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from output import print_sample
import matplotlib.pyplot as plt

def train_low_energy_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        printed = False

        for batch in train_loader:
            states = batch.states.to(device, non_blocking=True)  # [B, T, Ch, H, W]
            actions = batch.actions.to(device, non_blocking=True)  # [B, T-1, 2]
            if not printed:
                print_sample(states[60])
                printed = True

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

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        epoch_loss = 0.0

        count = 0
        for batch in train_loader:
            print(f'{count},',end="")
            count = count + 1
            observations = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]
            predicted_states, target_states = model(observations, actions)

            loss = model.loss(predicted_states, target_states)

            #if count == 10:
            #    plt.plot(predicted_states[0,0].detach().cpu().numpy(), label="Predicted")
            #    plt.plot(target_states[0,0].detach().cpu().numpy(), label="Target")
            #    plt.legend()
            #    plt.show()

            optimizer.zero_grad()
            loss.backward()
            #for name, param in model.named_parameters():
            #    print(f"{name}: {param.data.norm()} -> {param.data.norm()}")
            #for param in model.parameters():
            #    if param.grad is not None:
            #        print(f"Gradient norm: {param.grad.norm()}")
            #    else:
            #        print("No gradient computed for this parameter.")

            optimizer.step()
            epoch_loss += loss.item()
            print(f"Batch loss: {loss.item()}")


        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.10f}")

