import os
import argparse


def parse_args():
    argparser = argparse.ArgumentParser(description="DisCoNet PyTorch implementation")
    argparser.add_argument('--dataset', type=str, default='imagenet', help='dataset name', choices=['imagenet'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--latent_dim', type=int, default=128, help='latent dimension')
    argparser.add_argument('--hidden_dims', type=int, nargs='+', default=None, help='hidden dimensions')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--gen_weight', type=float, default=0.002, help='generator weight')
    argparser.add_argument('--recon_weight', type=float, default=0.002, help='reconstruction weight')
    argparser.add_argument('--sample_and_save_frequency', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--discriminator_checkpoint', type=str, default=None, help='discriminator checkpoint path')
    argparser.add_argument('--ood_task', type=str, default='near', help='ood task', choices=['near', 'far', 'covar'])
    argparser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    argparser.add_argument('--patches', type=int, default=16, help='number of patches')
    argparser.add_argument('--loss_type', type=str, default='mse', help='loss type', choices=['mse', 'ssim'])
    argparser.add_argument('--kld_weight', type=float, default=1e-4, help='kld weight')
    argparser.add_argument('--no_wandb', action='store_true', default=False, help='no wandb')

    return argparser.parse_args()
# EOF
