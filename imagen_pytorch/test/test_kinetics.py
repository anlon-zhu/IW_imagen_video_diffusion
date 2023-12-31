import os
import time
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, datasets
from imagen_pytorch import Unet, Imagen, ImagenTrainer

# Library by @lucidrains https://github.com/lucidrains/imagen-pytorch
# Verified with version 1.21.4

# Example script to train a conditional diffusion model on Kinetics at 32x32 resolution
# This exact file takes < 17Go of VRAM
# Training takes ~40 minutes on one A5000


class KineticsCond(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
        ])
        self.kinetics = datasets.Kinetics(
            root="data", train=train, download=True,
            transform=self.transform)

    def __len__(self):
        return len(self.kinetics)

    def __getitem__(self, i):
        img, label = self.kinetics[i]
        img = img.repeat(3, 1, 1)
        hot_label = torch.zeros(10)
        hot_label[label] = 1
        return img, hot_label.unsqueeze(0)


def delay2str(t):
    t = int(t)
    secs = t % 60
    mins = (t//60) % 60
    hours = (t//3600) % 24
    days = t//86400
    string = f"{secs}s"
    if mins:
        string = f"{mins}m {string}"
    if hours:
        string = f"{hours}h {string}"
    if days:
        string = f"{days}d {string}"
    return string


if __name__ == "__main__":

    experiment_path = os.path.join(
        "experiments", "conditional_kinetics_diffusion")
    images_path = os.path.join(experiment_path, "images")
    os.makedirs(images_path, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # Generate one-hot embedding for each digit
    emb_test = torch.nn.functional.one_hot(
        torch.arange(10)).float()[:, None, :]

    # Define model
    unet = Unet(
        dim=128,
        dim_mults=(1, 2, 4),
        num_resnet_blocks=3,
        layer_attns=(False, True, True),  # type: ignore
        layer_cross_attns=(False, True, True),  # type: ignore
        max_text_len=1,  # maximum number of embeddings per image
    )

    imagen = Imagen(
        unets=unet,
        image_sizes=32,
        text_embed_dim=10,  # dimension of one-hot embeddings
    )

    trainer = ImagenTrainer(
        imagen=imagen,
    ).to(device)

    # If you want to resume training from a checkpoint
    # trainer.load(path_to_checkpoint.pt)

    # Define dataset
    trainer.add_train_dataset(
        KineticsCond(train=True),
        batch_size=128, num_workers=16)
    trainer.add_valid_dataset(
        KineticsCond(train=False),
        batch_size=128, num_workers=16)

    # Trainning variables
    start_time = time.time()
    avg_loss = 1.0
    w_avg = 0.99
    target_loss = 0.005

    # Train
    print(f"Started training with target loss of {target_loss}")
    while avg_loss > target_loss:  # Should converge in < 5000 steps

        loss = trainer.train_step(unet_number=1)
        avg_loss = w_avg * avg_loss + (1 - w_avg) * loss

        print(
            f'Step: {trainer.steps.item():<6} | Loss: {loss:<6.4f} Avg Loss: {avg_loss:<6.4f} | {delay2str(time.time() - start_time):<10}',
            end='\r')  # type: ignore

        if trainer.steps % 500 == 0:  # type: ignore
            # Calculate validation loss
            valid_loss = np.mean(
                [trainer.valid_step(unet_number=1) for _ in range(10)])
            # type: ignore
            print(f'Step: {trainer.steps.item():<6} | Loss: {loss:<6.4f} Avg Loss: {avg_loss:<6.4f} | {delay2str(time.time() - start_time):<10} | Valid Loss: {valid_loss:<8.4f}')

            # Generate one image per class
            images = trainer.sample(
                batch_size=10, return_pil_images=True,
                text_embeds=emb_test, cond_scale=3.)  # returns List[Image]
            images = np.concatenate([np.array(img)
                                    for img in images], axis=1)
            Image.fromarray(images).save(
                os.path.join(
                    images_path,
                    f"sample-{str(trainer.steps.item()).zfill(10)}.png"))  # type: ignore

    # Final validation loss
    valid_loss = np.mean(
        [trainer.valid_step(unet_number=1) for _ in range(10)])
    print(f'Step: {trainer.steps.item():<6} | Loss: {loss:<6.4f} Avg Loss: {avg_loss:<6.4f} | {delay2str(time.time() - start_time):<10} | Valid Loss: {valid_loss:<8.4f}')  # type: ignore

    # Generate images
    images = trainer.sample(
        batch_size=10, return_pil_images=True, text_embeds=emb_test,
        cond_scale=3.)  # returns List[Image]
    images = np.concatenate([np.array(img) for img in images], axis=1)
    Image.fromarray(images).save(os.path.join(
        experiment_path, f"final_sample.png"))  # type: ignore

    # Save model
    trainer.save(os.path.join(experiment_path,
                 f"trained_kinetics.pt"))  # type: ignore

    print("Done!")
