from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
from train import train_low_energy_model
import torch
from models import MockModel
from models import LowEnergyOneModel
from models import LowEnergyTwoModel
import glob
import torch.optim as optim


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    #print(torch.cuda.memory_summary(abbreviated=True))
    return device

def load_training_data(device):
    data_path = "/scratch/DL24FA/train"
    #data_path = "/scratch/ph1499/partial"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}",
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

if __name__ == "__main__":

    num_epochs = 25
    learning_rate = 1e-4

    device = get_device()
    #model = LowEnergyOneModel(device=device).to(device)
    model = LowEnergyTwoModel(device=device).to(device)
    train_loader = load_training_data("cpu") # cpu first then gpu?

    train_low_energy_two_model(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    ####probe_train_ds, probe_val_ds = load_data(device)
    #model = load_model() # for empty model
    ####evaluate_model(device, model, probe_train_ds, probe_val_ds)

