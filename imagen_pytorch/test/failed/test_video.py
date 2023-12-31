import torch
from tqdm import tqdm
from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from video_data import Dataset, video_tensor_to_gif
# !pip install tensorflow-datasets
import tensorflow_datasets as tfds
import tensorflow as tf

# Construct a tf.data.Dataset
ds = tfds.load('mnist', split='train',
               as_supervised=True, shuffle_files=True)

# Build your input pipeline
ds = ds.shuffle(1000).batch(128).prefetch(10).take(5)
print("Loading Unet3D")
model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
)

unet1 = Unet3D(dim=128, channels=1, dim_mults=(1, 2, 4, 8)).cuda()
unet2 = Unet3D(dim=128, channels=1, dim_mults=(1, 2, 4, 8)).cuda()

print('Loading imagen...')
imagen = ElucidatedImagen(
    condition_on_text=False,
    unets=(unet1, unet2),
    ignore_time=False
    # channels=1,
    # image_sizes=(64, 128),
    # random_crop_sizes=(None, 16),
    # num_sample_steps=200,
    # cond_drop_prob=0.1,
    # sigma_min=0.002,
    # sigma_max=(80, 160),
    # sigma_data=0.5,
    # rho=7,
    # P_mean=-1.2,
    # P_std=1.2,
    # S_churn=80,
    # S_tmin=0.05,
    # S_tmax=50,
    # S_noise=1.003,
).cuda()

print('Loading trainer...')
trainer = ImagenTrainer(
    imagen=imagen,
    # whether to split the validation dataset from the training
    split_valid_from_train=True
).cuda()

print('Loading dataset...')
dataset = Dataset(folder='./data',
                  image_size=64, num_frames=32)
trainer.add_train_dataloader(dl, batch_size=1)
# trainer.add_train_dataset(dl, batch_size=1)

for i in tqdm(range(200000)):
    loss = trainer.train_step(unet_number=1, max_batch_size=4)
    print(f'loss: {loss}')

    if not (i % 500):
        valid_loss = trainer.valid_step(unet_number=1, max_batch_size=4)
        print(f'valid loss: {valid_loss}')

# sample
# extrapolating to 20 frames from training on 10 frames
videos = trainer.sample(video_frames=20)

print('Shape of sample:', videos.shape)  # (4, 3, 20, 32, 32)

for i, video in enumerate(videos):
    video_tensor_to_gif(video, f'sample{i}.gif')
