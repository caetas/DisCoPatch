# DisCoPatch

DisCoPatch is a generative model that combines the power of Variational Autoencoders (VAEs) with adversarial training. VAEs are a type of deep generative model that can learn to generate new data samples by capturing the underlying distribution of the training data. Adversarial training, on the other hand, involves training a discriminator network to distinguish between real and generated samples, while simultaneously training the generator network to fool the discriminator.

## Parameters

<center>

| Argument                      | Description                                        | Default    | Choices                          |
|-------------------------------|----------------------------------------------------|------------|----------------------------------|
| `--dataset`                   | Dataset name                                       | `imagenet` | `imagenet`                       |
| `--batch_size`                | Batch size                                         | `128`      |                                  |
| `--n_epochs`                  | Number of epochs                                   | `100`      |                                  |
| `--lr`                        | Learning rate                                      | `0.0002`   |                                  |
| `--latent_dim`                | Latent dimension                                   | `128`      |                                  |
| `--hidden_dims`               | Hidden dimensions                                  | `None`     |                                  |
| `--checkpoint`                | Checkpoint path                                    | `None`     |                                  |
| `--num_samples`               | Number of samples                                  | `16`       |                                  |
| `--gen_weight`                | Generator weight                                   | `0.002`    |                                  |
| `--recon_weight`              | Reconstruction weight                              | `0.002`    |                                  |
| `--sample_and_save_frequency` | Sample and save frequency                          | `5`        |                                  |
| `--discriminator_checkpoint`  | Discriminator checkpoint path                      | `None`     |                                  |
| `--ood_task`                  | Type of OOD detection task                         | `near`     | `near`, `far`, `covar`           |
| `--num_workers`               | Number of dataloader workers                       | `0`        |                                  |
| `--patches`                   | Number of patches per image                        | `16`       |                                  |
| `--no_wandb`                  | Disable wandb logs                                 | `False`    |                                  |

</center>

You can find out more about the parameters by checking [`util.py`](./../src/discopatch/utils/util.py) or by running the following command on the example script:

    python train_discopatch.py --help

## Training

To replicate the experiments performed in the paper, please use the following commands:

**ImageNet-1K**

    python train_discopatch.py --dataset imagenet --batch_size 67 --patches 48 --hidden_dims 128 256 512 1024 --latent_dim 1024 --n_epochs 80 --lr 8.5e-5 --gen_weight 1e-3 --recon_weight 1e-3 --sample_and_save_freq 2 --num_workers 8

## OOD Detection

To perform OOD detection you must indicate your ID dataset, the type of OOD detection task you want to perform and provide the discriminator checkpoint:

    python eval_discopatch.py --ood_task near --patches 64 --latent_dim 1024 --hidden_dims 128 256 512 1024 --discriminator_checkpoint ../../models/DisCoPatch/Discriminator_imagenet.pt