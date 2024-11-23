from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
from models import LowEnergyOneModel
import glob
import torch.optim as optim


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_training_data(device):
    data_path = "/scratch/DL24FA"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )

    return train_ds

def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    model = MockModel()
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")

'''
if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
'''

if __name__ == "__main__":

    print('step 1')
    num_epochs = 25
    learning_rate = 1e-4
    batch_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device {device}')

    model = LowEnergyOneModel(device=device).to(device)
    train_loader = load_training_data("cpu")

    train_low_energy_one_model(
        model=model,
        train_loader=train_loader,
        num_epochs=50,
        learning_rate=1e-4,
        device=device,
    )

    '''
    model = LowEnergyOneModel(device=device).to(device)
    print('model created')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('optimizer created')

    train_loader = load_training_data(device)
    print('loader ready')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f'Epoch {epoch+1}')
    
        for batch in train_loader:
            print('X', end="")
            states = batch.states.to(device, non_blocking=True)  # [B, T, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T-1, action_dim]

            print('O', end="")
    
            positive_energy = model(states, actions)
            shuffled_actions = actions[torch.randperm(actions.size(0))] 
            negative_energy = model(states, shuffled_actions)
    
            loss = model.loss(positive_energy, negative_energy)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")
        '''

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

    
    '''
        B, T, Ch, H, W = 8, 10, 3, 64, 64
        action_dim = 2
    
        model = LowEnergyOneModel(device="cuda").to("cuda")
        states = torch.rand(B, T, Ch, H, W).to("cuda")
        actions = torch.rand(B, T - 1, action_dim).to("cuda")
    
        positive_energy = model(states, actions)
        negative_energy = model(states, torch.roll(actions, shifts=1, dims=1)) 
    
        loss = model.loss(positive_energy, negative_energy)
        '''
