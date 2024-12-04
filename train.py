import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from output import print_sample
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda", test_mode=False):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(num_epochs * len(train_loader)))
    for epoch in tqdm(range(num_epochs)):

        if epoch == 2:  # Freeze target_encoder after the 2nd epoch
            for param in model.target_encoder.parameters():
                param.requires_grad = False
            
            # Update the optimizer to exclude the frozen parameters
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=learning_rate
            )
    
        model.train()
        epoch_loss = 0.0


        for batch in train_loader:
            states = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]

            predicted_states, target_states = model(states, actions)
            loss = model.loss(predicted_states, target_states)
            optimizer.zero_grad()
            loss.backward()


            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.update(1)


        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.10f}")
    return predicted_states, target_states

