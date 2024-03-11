# Import of libraries
import random
import numpy as np
import os
import pandas as pd
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Tuple, TypeVar
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


no_train = False
batch_size = 128
n_epochs = 50
lr = 0.0001

store_path = "ddpm_otdr.pt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df = pd.read_csv('OTDR_diffusion.csv')

#%%
# Group the data by 'Class'
grouped = df.groupby('Class')

# Calculate statistics for each class
class_statistics = {}
for name, group in grouped:
    statistics = {
        'Class': name,
        #'SNR_Min': group['SNR'].min(),
        #'SNR_Max': group['SNR'].max(),
        #'SNR_Mean': group['SNR'].mean(),
        #'SNR_Median': group['SNR'].median(),
        #'SNR_StdDev': group['SNR'].std(),
        'Amp_Min': group['Max_Amplitude'].min(),
        'Amp_Max': group['Max_Amplitude'].max(),
        'Amp_Mean': group['Max_Amplitude'].mean(),
        'Amp_Median': group['Max_Amplitude'].median(),
        'Amp_StdDev': group['Max_Amplitude'].std(),
    }
    class_statistics[name] = statistics

# Convert the statistics to a DataFrame
statistics_df = pd.DataFrame(list(class_statistics.values()))

# Print or analyze the statistics as needed
print(statistics_df)
#%%
def show_signals(signals, title=""):
    """Shows the provided signals as sub-pictures in a square"""

    # Converting signals to CPU numpy arrays
    if type(signals) is torch.Tensor:
        signals = signals.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(signals) ** (1 / 2))
    cols = round(len(signals) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax= fig.add_subplot(rows, cols, idx + 1)

            if idx < len(signals):
                ax.plot(signals[idx])
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()
#%%
def show_first_batch(loader):
    for batch in loader:
        show_signals(batch[0], "Images in the first batch")
        break
#%%
# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
class TraceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, traces, labels, levels, amps):
        'Initialization'
        self.list_IDs = list_IDs
        self.traces = traces
        self.labels = labels
        self.levels = levels
        self.amps = amps

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.traces[index]
        y = self.labels[index]
        z = self.levels[index]
        w = self.amps[index]

        return X, y, z, w
#%%

# Assuming df is your DataFrame containing signals
data=df[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30']]
# Compute number of total traces
n_data = data.shape[0]
df['SNR']=df['SNR'].round()

labels_dict = dict()
levels_dict = dict()
amps_dict = dict()
idx = np.arange(n_data)

labels = torch.tensor(df['Class'].values)
levels = torch.tensor(df['SNR'].values)
amps = torch.tensor(df['Max_Amplitude'].values)
data=torch.tensor(data.values)


#%%
# Associate IDs to labels
for i in range(n_data):
    new_dict = {idx[i]: labels[i]}
    labels_dict.update(new_dict)

# Associate IDs to levels
for i in range(n_data):
    new_dict = {idx[i]: levels[i]}
    levels_dict.update(new_dict)

# Associate IDs to amps
for i in range(n_data):
    new_dict = {idx[i]: amps[i]}
    amps_dict.update(new_dict)

#%%
signal_dataset = TraceDataset(idx, data, labels_dict, levels_dict, amps_dict)
# Assuming 'dataset' is an instance of your dataset class
#print(f"Total number of labels: {len(signal_dataset.labels)}")
#print(f"Reported dataset size: {len(signal_dataset)}")

# Assert to ensure they match
#assert len(signal_dataset.labels) == len(signal_dataset), "Mismatch between the number of labels and the reported dataset size."
loader = DataLoader(signal_dataset, batch_size, shuffle=True)
#%%
show_first_batch(loader)
#%%
# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

# DDPM class
T = TypeVar("T", bound=torch.Tensor)

def broadcast_like(z: T, like: Optional[torch.Tensor]) -> T:
    """
    Add broadcast dimensions to x so that it can be broadcast over ``like``
    """
    if like is None:
        return z
    return z[(...,) + (None,) * (like.ndim - z.ndim)]

#%%
class DDPM(torch.nn.Module):
    def __init__(self, model, N=1000, noise_schedule_type: str = "linear", device=device):
        super(DDPM, self).__init__()
        self.N = N
        self.device = device
        self.model = model.to(device)

        self._betas, self._alphas, self._alpha_bars = (
            torch.nn.Parameter(x, requires_grad=False)
            for x in self.get_coefs(N, noise_schedule_type)
        )

    @staticmethod
    def get_coefs(
        N: int, type: str= "linear"
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Get the coefficients for the noise schedule.

        Args:
            N (int): number of steps in the diffusion process.
            type (str, optional): type of noise schedule. Defaults to "linear".

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                betas, alphas, alpha_bars
        """
        if type == "linear":
            # setting the betas to be linearly spaced between beta_min=0.0001 and beta_max=0.02
            # alpha_t = 1 - beta_t
            # alpha_bar_t = prod_s=1^t (alpha_s)
            beta_min = 0.0001
            beta_max = 0.02
            betas = torch.linspace(beta_min, beta_max, N).to(device)
            alphas = 1 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
        elif type == "cosine":
            # setting alpha_bars using the cosine schedule, and then computing alphas and betas
            # accordingly
            def cos2(t, s=0.001):
                return torch.cos((t / N + s) / (1 + s) * np.pi / 2) ** 2

            alpha_bars = (cos2(torch.arange(N)) / cos2(torch.zeros(1))).to(device)
            alphas = torch.cat([alpha_bars[:1], alpha_bars[1:] / alpha_bars[:-1]]).to(device)
            betas = 1 - alphas

            assert torch.allclose(alpha_bars, torch.cumprod(alphas, dim=0))

        return betas, alphas, alpha_bars

    def betas(self, t: torch.LongTensor) -> torch.FloatTensor:
        return self._betas[t].to(device)

    def alphas(self, t: torch.LongTensor) -> torch.FloatTensor:
        return self._alphas[t].to(device)

    def alpha_bars(self, t: torch.LongTensor) -> torch.FloatTensor:
        return self._alpha_bars[t].to(device)

    def _q_mean(
        self,
        z: torch.FloatTensor,
        t: torch.LongTensor
    ) -> torch.FloatTensor:
        # mean of the distribution q(z_t | z_0)
        z = z.squeeze()
        sqrt_alpha_bars = broadcast_like(torch.sqrt(self.alpha_bars(t)), z).to(device)
        return (sqrt_alpha_bars * z).to(device)

    def _q_std(
        self,
        z: torch.FloatTensor,
        t: torch.LongTensor
    ) -> torch.FloatTensor:

        z = z.squeeze()
        # std of the distribution q(z_t | z_0)
        std = torch.sqrt(1.0 - self.alpha_bars(t))
        return broadcast_like(std, z).to(device)

    def q_sample(
        self,
        z: torch.FloatTensor,
        t: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Sample from q(z_t | z_0)

        mean = self._q_mean(z, t).to(device)
        std = self._q_std(z, t).to(device)
        epsilon = torch.randn_like(z)

        z_t = epsilon * std + mean
        z_t = z_t.to(device)

        return z_t, epsilon
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
    ) -> np.ndarray:

        # Sample from the prior N(0, I)
        z_t = torch.randn(shape, device=self.device)

        # Iterate from t = N-1 to 1
        for t_discrete in reversed(range(1, self.N)):
            t_discrete = torch.ones(shape[0], device=self.device).long() * t_discrete
            mean = self._p_mean(z_t, t_discrete)
            std = self._p_std(z_t, t_discrete)
            # sample z_{t-1} | z_t
            epsilon = torch.randn_like(z_t)
            z_t = epsilon * std + mean

        return self.sample_x0_given_x1(z_t).cpu().numpy()

    def _p_mean(
        self,
        z: torch.FloatTensor,
        t: torch.LongTensor,
        labels: torch.LongTensor,
        levels: torch.LongTensor,
        amps: torch.LongTensor,
    ) -> torch.FloatTensor:
        # mean of the distribution p(z_{t-1} | z_t)
        beta = broadcast_like(self.betas(t), z)
        alpha = broadcast_like(self.alphas(t), z)
        alpha_bar = broadcast_like(self.alpha_bars(t), z)
        #z = z.squeeze()
        #z = z.unsqueeze(0)
        #labels = labels.unsqueeze(0)
        #levels = levels.unsqueeze(0)
        #amps = amps.unsqueeze(0)
        #print(z.shape)
        #print(labels.shape)
        #print(levels.shape)
        #print(amps.shape)
        epsilon_pred = self.model(
            z, t.float() / self.N, labels, levels, amps,
        ).to(device)
        #print("eps", epsilon_pred.shape)

        mean = (z - beta * epsilon_pred / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha)
        return mean.to(device)

    def _p_std(
        self,
        z: torch.LongTensor,
        t: torch.LongTensor
    ) -> torch.FloatTensor:

        z = z.squeeze()
        beta = broadcast_like(self.betas(t), z)
        return torch.sqrt(beta)

    def p_sample(self, z_t, t, labels, levels, amps, device):
        """
        Sample from p(z_{t-1} | z_t) using the provided _p_mean and _p_std functions.

        Parameters:
        - z_t: torch.FloatTensor, the current noisy signal at time t.
        - t: torch.LongTensor, the current time step.
        - device: The device on which to perform the calculations (e.g., 'cuda' or 'cpu').

        Returns:
        - z_prev: torch.FloatTensor, the denoised signal at time t-1.
        """
        # Ensure the inputs are on the correct device
        z_t = z_t.to(device)
        t = t.to(device)

        # Calculate the mean and standard deviation for p(z_{t-1} | z_t)
        mean = self._p_mean(z_t, t, labels, levels, amps).to(device)
        std = self._p_std(z_t, t).to(device)

        # Sample epsilon (noise) from a standard normal distribution
        epsilon = torch.randn_like(z_t, device=device)
        z_prev = mean + epsilon * std
        #print(z_prev.shape)
        return z_prev.to(device)

    def forward(self, z, t, eta=None):
        # Make input signal more noisy
        t=t.unsqueeze(1)
        a_bar = broadcast_like(self.alpha_bars(t),z)

        if eta is None:
            eta = torch.randn(z).to(self.device)

        noisy = broadcast_like(torch.sqrt(a_bar), z) * z + broadcast_like(torch.sqrt(1 - a_bar), z) * eta

        return noisy

    def p_sample_full_reverse(self, z_t, t, labels, levels, amps, device):
        """
        Perform the full reverse sampling process to denoise the signal.

        Parameters:
        - z_t: torch.FloatTensor, the noisy signal at the final timestep.
        - labels, levels, amps: Conditioning information.
        - device: The device on which to perform the calculations (e.g., 'cuda' or 'cpu').

        Returns:
        - z_0: torch.FloatTensor, the reconstructed original signal.
        """
        # Ensure the inputs are on the correct device
        z_t = z_t.to(device)

        # Iterate over the timesteps in reverse order
        for t in reversed(range(self.N)):
            t_tensor = torch.tensor([t], dtype=torch.long, device=device)

            # Perform a single reverse step
            z_t = self.p_sample(z_t, t_tensor, labels, levels, amps, device)

        # After completing the reverse steps, z_t should be close to the original signal (z_0)
        return z_t

    def backward(self, x, t, labels, snr, amps, device):
        # The network returns its estimation of the noise that was added.
        return self.model(x, t, labels, snr, amps).to(device)

#%%
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, feature_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.positional_embeddings = self.create_positional_embeddings()

    def create_positional_embeddings(self):
        # Create a matrix of shape [sequence_length, feature_dim]
        position = torch.arange(self.sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2).float() * -(math.log(10000.0) / self.feature_dim))

        positional_embeddings = torch.zeros((self.sequence_length, self.feature_dim))
        positional_embeddings[:, 0::2] = torch.sin(position * div_term)
        positional_embeddings[:, 1::2] = torch.cos(position * div_term)

        return positional_embeddings

    def forward(self, x):
        # Assume x is of shape [batch_size, sequence_length]
        # Expand positional embeddings to match batch size and add them to input
        batch_size = x.size(0)
        positional_embeddings = self.positional_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return positional_embeddings

class ScoreModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 50
        self.sequence_length = 30  # Assuming sequence_length is 30
        self.pos_embedding = SinusoidalPositionalEmbedding(self.sequence_length, self.embedding_dim)

        self.label_embedding = nn.Embedding(6, 10)
        self.level_embedding = nn.Embedding(31, 90)
        self.amp_embedding = nn.Embedding(21, 50)

        self.fc_in = torch.nn.Linear(180, 128)
        self.gru1 = torch.nn.GRU(128, 256, num_layers=1,
                                 batch_first=True, bidirectional=False)
        self.dropout = torch.nn.Dropout(0.2)
        self.gru2 = torch.nn.GRU(256, 128, num_layers=1,
                                 batch_first=True, bidirectional=False)
        self.fc_out = torch.nn.Linear(128, 30)

    def forward(self, z_t, t, labels, levels, amps):
        batch_size = z_t.size(0)
        #z_t = z_t.squeeze(0)
        #pos_embeddings = self.pos_embedding(t).to(device)

        labels_emb = self.label_embedding(labels).to(device)
        snr_emb = self.level_embedding(levels).to(device)
        amps_emb = self.amp_embedding(amps).to(device)

        #print(f"labels_emb shape: {labels_emb.shape}")
        #print(f"snr_emb shape: {snr_emb.shape}")
        #print(f"amps_emb shape: {amps_emb.shape}")
        #labels_emb = labels_emb.expand(z_t.size(0), -1)
        #snr_emb = snr_emb.expand(z_t.size(0), -1)
        #amps_emb = amps_emb.expand(z_t.size(0), -1)

        #z_t_with_pos = z_t + pos_embeddings
        #z_t = z_t_with_pos.reshape(batch_size, -1)
        #print(z_t.shape)
        z_t = torch.cat([z_t, labels_emb, snr_emb, amps_emb], dim=-1)
        z_t = z_t.to(dtype=torch.float32).to(device)
        #z_t_augmented = z_t_augmented.to(device).to(dtype=torch.float32)
        x = F.leaky_relu(self.fc_in(z_t), 0.2)
        x, _ = self.gru1(x.unsqueeze(1))
        x = self.dropout(x)
        gru_output, _ = self.gru2(x)
        score = self.fc_out(gru_output.squeeze(1))
        return score
#%%
ddpm = DDPM(ScoreModel(), N=1000, noise_schedule_type = "linear", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#%%
def show_forward(ddpm, loader, device):
    # Initialize a variable to save the fully noisy signal information
    fully_noisy_signal_info = {}

    # Process just the first batch from the loader
    for (batch_traces, batch_labels, batch_snr, batch_amps) in loader:
        # Isolate the first signal and its conditions from the batch
        x = batch_traces[0].unsqueeze(0)  # Add batch dimension
        x_label = batch_labels[0]
        x_level = batch_snr[0].long()
        x_amp = batch_amps[0]

        # Prepare a title with condition labels for the signal
        title_base = f"Label: {x_label}, SNR: {x_level}, Amp: {x_amp}"
        show_signals(x.cpu(), f"Original - {title_base}")  # Show the original signal with conditions

        # Iterate through noise levels to generate and show noisy signals
        for percent in [0.25, 0.5, 0.75, 1]:
            noisy_signal, _ = ddpm.q_sample(
                x.to(device),
                torch.tensor([int(percent * ddpm.N) - 1]).to(device)
            )

            # Visualize each noisy signal
            show_signals(noisy_signal.cpu(), f"{int(percent * 100)}% - {title_base}")


        break

show_forward(ddpm, loader, device)
#%%
def save_forward(ddpm, loader, device):
    ddpm.eval()
    for (batch_traces, batch_labels, batch_snr, batch_amps) in loader:
        x = batch_traces[0].unsqueeze(0).to(device)  # Noisy signal
        x_label = batch_labels[0].to(device)
        x_level = batch_snr[0].long().to(device)
        x_amp = batch_amps[0].to(device)

        # Assuming the generation of the fully noisy signal is done here
        percent_noise = 1.0
        fully_noisy_signal, _ = ddpm.q_sample(
            x,
            torch.tensor([int(percent_noise * ddpm.N) - 1], device=device)
        )

        # Only return the fully noisy signal and its conditions
        return fully_noisy_signal, x_label, x_level, x_amp
#%%
def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    huber = torch.nn.HuberLoss(delta=1.0)
    best_loss = float("inf")
    N = ddpm.N
    ddpm=ddpm.to(device)
    start = time.process_time()
    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, (batch_traces, batch_labels, batch_snr, batch_amps) in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch_traces.float().to(device)
            x_label = batch_labels.long().to(device)
            #print(x_label.shape)
            x_snr = batch_snr.long().to(device)
            x_amp = batch_amps.long().to(device)
            n = len(x0)

            #print(x0.shape)
            #print(x_label.shape)
            #print(x_snr.shape)
            #print(x_amp.shape)

            # Picking some noise for each of the signals in the batch, a timestep and the respective alpha_bars
            #print(eta.shape)
            batch_size = n
            t = torch.randint(0, N, (n,)).to(device)
            betas_t = ddpm.betas(t).to(device)
            eta = torch.randn(batch_size, 30).float().to(device)* torch.sqrt(betas_t.unsqueeze(1))

            # Computing the noisy  based on x0 and the time-step (forward process)
            noisy_sigs = ddpm(x0, t, eta)
            # Getting model estimation of noise based on the signals and the time-step
            eta_theta = ddpm.backward(noisy_sigs, t.reshape(n, -1), x_label, x_snr, x_amp, device)
            loss = huber(eta_theta, eta)
            #print(eta_theta.shape)
            # Optimizing the MSE between the
            #loss = losses.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        #if display:
           #show_signals(generate_new_signals(ddpm, device=device), f"Signals generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " <-- Model Stored"

        print(log_string)

        print("Total time to train: ", time.process_time() - start)

# Training
store_path = "ddpm_otdr.pt"
if not no_train:
   training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

#%%
model=ScoreModel().to(device)
def show_backward(ddpm, loader, device):
    ddpm.eval()
    model.eval()
    # Use show_forward to get the noisiest signal and its conditions
    fully_noisy_signal, x_label, x_level, x_amp = save_forward(ddpm, loader, device)
    #fully_noisy_signal=fully_noisy_signal.squeeze(0)
    x_label=x_label.unsqueeze(0)
    x_level = x_level.unsqueeze(0)
    x_amp = x_amp.unsqueeze(0)
    # Now, use the fully noisy signal in the denoising process
    title_base = f"Label: {x_label.item()}, SNR: {x_level.item()}, Amp: {x_amp.item()}"
    show_signals(fully_noisy_signal.cpu(), f"Noisy Signal - {title_base}")
    for percent in [1, 0.75, 0.5, 0.25, 0]:
        timestep = torch.tensor([int(percent * ddpm.N) - 1], device=device)
        #print(timestep.shape)
        #print(fully_noisy_signal.shape)
        #print(x_label.shape)
        #print(x_level.shape)
        #print(x_amp.shape)
        denoised_signal = ddpm.p_sample(
            fully_noisy_signal,
            timestep,
            x_label,
            x_level,
            x_amp,
            device=device
        )
        #print(denoised_signal.shape)
        #denoised_signal=denoised_signal.squeeze(0)
        show_signals(denoised_signal.cpu(), f"{int(percent * 100)}% - {title_base}")


#show_backward(ddpm, loader, device)
#%%
best_model = DDPM(ScoreModel(), N=1000, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")
#%%
# Assuming 'dataloader' is your DataLoader instance
# Fetch one batch of data
batch = next(iter(loader))
x_original, labels, levels, amps = batch

# Assuming the first dimension of x_original is the batch dimension
x_original = x_original[0:1]  # Selects the first signal in the batch, keeps batch dimension for consistency

# Move the signal to the same device as your model
x_original = x_original.to(device)
labels = labels.long().to(device)
levels = levels.long().to(device)
amps = amps.long().to(device)

# Simulate a noisy version of the normalized signal for a specific timestep
t_index = best_model.N-1  # Example timestep index
beta_t = best_model.betas(t_index)  # Get beta value at the specified timestep
alpha_t = 1.0-beta_t

noise = torch.randn_like(x_original) * torch.sqrt(beta_t.clone().detach()).to(device)
x_noisy = torch.sqrt(alpha_t).to(device) * x_original + noise

# With the model in eval mode, perform the backward operation to predict the noise
with torch.no_grad():
    # Assuming 'ddpm.backward' expects the inputs in a specific format, adjust accordingly
    predicted_noise = ddpm.backward(x_noisy, torch.tensor([t_index]), labels[0:1], levels[0:1], amps[0:1], device)

# This comparison is simplistic; in practice, you'd assess how closely predicted_noise matches the actual noise
noise_diff = (predicted_noise - noise).abs().mean()
print(f"Mean absolute difference between predicted and actual noise: {noise_diff.item()}")

# Plotting the signals for visualization
plt.figure(figsize=(15, 5))
plt.plot(x_original.cpu().squeeze(), label='Original Signal')
plt.plot(x_noisy.cpu().squeeze(), label='Noisy Signal')
plt.plot((x_noisy - predicted_noise).cpu().squeeze(), label='Predicted Denoised Signal')
plt.legend()
plt.show()


#%%

def generate_specific_class_signals(ddpm, device=None, sequence_length=30, N=None):
    """Generate new signals for classes 4 and 5 with specific SNR and sample counts, matching the noise addition and reduction process of the single-sample test."""
    with torch.no_grad():
        device = device or ddpm.device
        N = N or ddpm.N  # Use N from the model if not specified

        # Define samples for class 4 and 5
        n_samples_class_4 = 5256
        n_samples_class_5 = 4036

        # Prepare labels, SNR (levels), and max_amplitude (amps) tensors
        labels_class_4 = torch.full((n_samples_class_4,), 4, dtype=torch.long, device=device)
        labels_class_5 = torch.full((n_samples_class_5,), 5, dtype=torch.long, device=device)

        # Assuming SNR and Amps follow a specific distribution, adjust as needed
        levels_class_4 = torch.randint(20, 31, (n_samples_class_4,), device=device)
        levels_class_5 = torch.randint(20, 31, (n_samples_class_5,), device=device)
        amps_class_4 = torch.randint(0, 13, (n_samples_class_4,), device=device)
        amps_class_5 = torch.randint(0, 13, (n_samples_class_5,), device=device)

        df_data = []

        # Process each class separately
        for class_label, n_samples, labels, levels, amps in [
            (4, n_samples_class_4, labels_class_4, levels_class_4, amps_class_4),
            (5, n_samples_class_5, labels_class_5, levels_class_5, amps_class_5)]:
            x = torch.randn(n_samples, sequence_length, device=device)
            for t_index in reversed(range(N)):
                t_tensor = torch.full((n_samples,), t_index, dtype=torch.long, device=device)
                beta_t = ddpm.betas(t_index).to(device)
                alpha_t = 1 - beta_t

                noise = torch.randn_like(x) * torch.sqrt(beta_t)
                x_noisy = torch.sqrt(alpha_t) * x + noise
                predicted_noise = ddpm.backward(x_noisy, t_tensor.unsqueeze(1), labels, levels, amps, device)

                x = (x_noisy - predicted_noise)

                # Normalize signals to range [0, 1] right before appending to df_data
                x_min = x.min(dim=1, keepdim=True)[0]
                x_max = x.max(dim=1, keepdim=True)[0]
                x = (x - x_min) / (x_max - x_min)

            for i in range(n_samples):
                signal = x[i].cpu().numpy()
                df_data.append({
                    **{f'P{j + 1}': signal[j] for j in range(sequence_length)},
                    'Class': class_label,
                    'SNR': levels[i].item(),
                    'Max_Amplitude': amps[i].item()
                })

        df_generated = pd.DataFrame(df_data)

    return df_generated


print("Generating new signals")
generated = generate_specific_class_signals(
        best_model,
        device=device,
        sequence_length=30
    )

#generated['Class'] = generated['Class'].replace(4, 4)
#generated['Class'] = generated['Class'].replace(5, 0)

columns_to_clip = [f'P{i}' for i in range(1, 30)]
generated[columns_to_clip] = generated[columns_to_clip].clip(lower=0, upper=1)

generated.to_csv('generated_data.csv', index=False)







