import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import warnings

#%%
# Turn warnings into exceptions
warnings.filterwarnings('error', category=UserWarning, message='Initializing zero-element tensors is a no-op')

# Your code here, e.g., model initialization, training loop, etc.
try:
    # Potentially problematic code that might trigger the warning
    pass
except UserWarning as e:
    print(e)
    # Use a debugger here or print a traceback to find the exact line
    import traceback
    traceback.print_exc()
#%%

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df = pd.read_csv('OTDR_diffusion.csv')

data=df[['SNR','P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30','Max_Amplitude']]

labels = torch.tensor(df['Class'].values)
df_signals=torch.tensor(data.values)

#%%

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

timesteps = 500

# compute betas
betas = linear_beta_schedule(timesteps=timesteps)

# compute alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# calculations for the forward diffusion q(x_t | x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

#%%

class NormalizeSignal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x * 2) - 1

#%%
transform = nn.Sequential(
     NormalizeSignal(),
)

signal = df_signals[0]
signal = signal.unsqueeze(0)
x_start = transform(signal)

#%%
def reverse_transform(signal):
    # Assuming the signal values are normalized between -1 and 1, scale them back
    signal = (signal + 1) / 2  # Scale to [0, 1]
    signal = signal * 255
    return signal

reverse_transform(x_start.squeeze())

#%%
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

#%%
# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start) # z (it does not depend on t!)

    # adjust the shape
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

"""Let's test on a specific time step, $t=20$:"""
#%%
def get_noisy_signal(x_start, t):

  x_noisy = q_sample(x_start, t=t) # add noise
  noisy_signal = reverse_transform(x_noisy.squeeze()) # turn back into PIL image

  return noisy_signal

#%%

t = torch.tensor([19])
diffused_signal=get_noisy_signal(signal[0], t)

#%%
# Convert the diffused signal back to a NumPy array for plotting
signal_np = signal.squeeze().numpy()
diffused_signal_np = diffused_signal.squeeze().numpy()  # Remove batch dimension and convert to NumPy

# Plotting the original and diffused signal
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(signal_np, label='Original Signal')
plt.title("Original Signal")
plt.xlabel("Step")
plt.ylabel("Value")

plt.subplot(1, 2, 2)
plt.plot(diffused_signal_np, label='Diffused Signal', color='red')
plt.title("Diffused Signal")
plt.xlabel("Step")
plt.ylabel("Value")

plt.tight_layout()
plt.show()

#%%

def plot_signals(signals, with_orig=False, row_title=None, **plot_kwargs):
    # Adjusting for the possibility of a single signal input
    if not isinstance(signals[0], list):
        signals = [signals]  # Make a 2d grid even if there's just 1 row

    num_rows = len(signals)
    num_cols = len(signals[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3, num_rows * 3), squeeze=False)

    for row_idx, row in enumerate(signals):
        if with_orig:
            row = [orig_signal] + row  # Assume orig_signal is defined or passed to the function
        for col_idx, signal in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.plot(signal, **plot_kwargs)
            ax.set(title=f'Timestep {col_idx}' if col_idx > 0 or not with_orig else 'Original')

    if row_title is not None:
        for row_idx, title in enumerate(row_title):
            axs[row_idx, 0].set_ylabel(title)

    plt.tight_layout()
    plt.show()


#%%
orig_signal = signal[0]  # For example, the original signal to compare with
timesteps = [1, 20, 50, 100, 199]
diffused_signals = [get_noisy_signal(orig_signal, torch.tensor([t])) for t in timesteps]

plot_signals(diffused_signals, with_orig=True, row_title=['Diffusion Process'])
#%%
# Let's look at how the time embeddings look like
from scripts.unet import SinusoidalPositionEmbeddings

time_emb = SinusoidalPositionEmbeddings(100)
t1 = time_emb(torch.tensor([10]))
t2 = time_emb(torch.tensor([12]))
t3 = time_emb(torch.tensor([30]))

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(t1.numpy()[0], label='t=10')
ax.plot(t2.numpy()[0], label='t=12')
ax.plot(t3.numpy()[0], label='t=30')
plt.legend()
plt.show()

#%%
try:
    from scripts.unet import Unet

    temp_model = Unet(
        dim=1,
        channels=1,
        dim_mults=(1, 2, 4,)
    )

    x_start = x_start.squeeze()
    x_start = x_start.unsqueeze(0).unsqueeze(0)
    x_start = x_start.float()
    #print(x_start)
    #temp_model.float()  # Convert all model parameters and buffers to double
    with torch.no_grad():
        out = temp_model(x_start, torch.tensor([40]))

    print(f"input shape: {x_start.shape}, output shape: {out.shape}")
except Exception as e:
    print(f"An error occurred: {e}")


#%%

def p_losses(denoise_model, x_start, t, loss_type="huber"):

    # random sample z
    noise = torch.randn_like(x_start)

    # compute x_t
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # recover z from x_t with the NN
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

#%%

# calculations for posterior q(x_{t-1} | x_t, x_0) = q(x_{t-1} | t, x_0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # β_t

@torch.no_grad()
def p_sample(model, x, t, t_index):

    # adjust shapes
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Use the NN to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    # Draw the next sample
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)  # beta_t
        noise = torch.randn_like(x)                                     # z
        return model_mean + torch.sqrt(posterior_variance_t) * noise    # x_{t-1}

#%%

# Sampling loop
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((shape[0],), i, device=device, dtype=torch.long), i)
        imgs.append(img)
    return imgs

#%%

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

#%%
transform = Lambda(lambda t: (t * 2) - 1)

#%%
class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, signals, transform=None):
        """
        signals: A DataFrame, array, or tensor where each row is a signal
        transform: A function/transform to apply to each signal
        """
        # Convert signals to a tensor if it's a DataFrame or array
        if isinstance(signals, pd.DataFrame):
            self.signals = torch.tensor(signals.values.astype(float), dtype=torch.float)
        elif isinstance(signals, np.ndarray):
            self.signals = torch.tensor(signals.astype(float), dtype=torch.float)
        else:
            self.signals = signals  # Assume signals is already a tensor

        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]  # Direct tensor indexing

        signal = signal.unsqueeze(0)

        if self.transform:
            signal = self.transform(signal)

        return signal

# Assuming df_signals is your DataFrame containing signals
signal_dataset = SignalDataset(df_signals, transform=transform)
#%%
# Create DataLoader for your signal dataset
batch_size = 64  # Adjust batch size as needed
shuffle = True  # Shuffle data each epoch
signal_dataloader = DataLoader(signal_dataset, batch_size=batch_size, shuffle=shuffle)

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=1,
    channels=1,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
#%%

epochs = 10

for epoch in range(epochs):
    for step, batch in enumerate(signal_dataloader):
      optimizer.zero_grad()

      batch_size = batch["pixel_values"].shape[0]
      batch = batch["pixel_values"].to(device)

      # sample t from U(0,T)
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = p_losses(model, batch, t)

      if step % 100 == 0:
        print(f"Epoch: {epoch}, step: {step} -- Loss: {loss.item():.3f}")

      loss.backward()
      optimizer.step()

#%%











