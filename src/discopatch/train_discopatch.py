from data.Dataloaders import pick_dataset
from models.DisCoPatch import DisCoPatch
from utils.util import parse_args
import wandb

if __name__ == '__main__':

    args = parse_args()

    if args.dataset == 'imagenet':
        img_size = 256
        channels = 3

    # Initialize wandb
    wandb.init(project='DisCoPatch',
                config={
                    'dataset': args.dataset,
                    'batch_size': args.batch_size,
                    'n_epochs': args.n_epochs,
                    'latent_dim': args.latent_dim,
                    'hidden_dims': args.hidden_dims,
                    'lr': args.lr,
                    'gen_weight': args.gen_weight,
                    'recon_weight': args.recon_weight,
                    'sample_and_save_frequency': args.sample_and_save_frequency,
                    'patches': args.patches,
                    'loss_type': args.loss_type,
                    'kld_weight': args.kld_weight,
                },
                name = 'DisCoPatch_{}'.format(args.dataset))

    # Load dataset, initialize model and train
    train_dataloader = pick_dataset(name = args.dataset, train=True, batch_size=args.batch_size, img_size=img_size, num_workers=args.num_workers, patches=args.patches)
    model = DisCoPatch(input_shape = img_size//4, input_channels = channels, args = args)
    model.train_model(train_dataloader, train_dataloader)

    # Finish wandb
    wandb.finish()