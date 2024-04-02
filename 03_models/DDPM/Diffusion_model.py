import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Define a class for the diffusion process
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        # Initialization of parameters for the diffusion model
        self.noise_steps = noise_steps  # Number of diffusion steps
        self.beta_start = beta_start    # Starting value of beta (variance schedule)
        self.beta_end = beta_end        # Ending value of beta (variance schedule)
        self.img_size = img_size        # Image size (assuming square images)
        self.device = device            # Device to run the model on (e.g., "cuda" or "cpu")

        # Prepare the noise schedule based on beta values and transfer it to the specified device
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta                     # Alpha values calculated from beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # Cumulative product of alpha values

    def prepare_noise_schedule(self):
        # Generates a linear schedule for beta values from beta_start to beta_end
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        # Apply noise to images based on the timestep t
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)  # Generate random noise
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ  # Return the noised image and the noise

    def sample_timesteps(self, n):
        # Sample random timesteps for each image in the batch
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        # Generate new images using the reverse diffusion process
        print(f"Sampling {n} new images....")
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)  # Start with random noise
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):  # Reverse diffusion process
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)  # Predict the noise to subtract
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)  # Add random noise in intermediate steps
                else:
                    noise = torch.zeros_like(x)  # No noise in the final step
                # Reverse diffusion equation to denoise the image
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()  # Set the model back to training mode
        # Normalize the images to have values between 0 and 1
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x - torch.amin(x))/(torch.amax(x) - torch.amin(x))
        return x
