from data.Dataloaders import *
from models.DisCoPatch import DisCoPatch
from utils.util import parse_args
import torch
import os
from config import data_raw_dir

if __name__ == '__main__':

    args = parse_args()
    args.batch_size = 32

    if args.dataset == 'imagenet':
        img_size = 256
        channels = 3
        near_ood = ['ninco', 'ssb-hard']
        far_ood = ['inaturalist', 'dtd', 'openimageo']
        in_loader = pick_dataset(name = args.dataset, train=False, batch_size=args.batch_size, img_size=img_size, num_workers=args.num_workers, patches=args.patches)
        if os.path.exists(os.path.join(data_raw_dir, 'ImageNet-C')):
            corruptions = os.listdir(os.path.join(data_raw_dir, 'ImageNet-C'))

    # Initialize model and load checkpoint
    model = DisCoPatch(input_channels=channels, input_shape=img_size//4, args=args)

    if args.discriminator_checkpoint is not None:
        model.discriminator.load_state_dict(torch.load(args.discriminator_checkpoint, weights_only=False), strict=False)
        model.eval()

    in_array = None

    # Evaluate OOD detection
    if args.ood_task == 'near':
        print(f"Near OOD Detection for {args.dataset}\n")
        for ood in near_ood:
            out_loader = pick_dataset(name = ood, train=False, batch_size=args.batch_size, img_size=img_size, num_workers=args.num_workers, patches=args.patches)
            auroc, fpr95, in_array, _ = model.outlier_detection(in_loader, out_loader, display=False, in_array=in_array, patches=args.patches)
            print(f"OOD: {ood}\nAUROC: {auroc:.4f}\nFPR95: {fpr95:.4f}\n\n")

    elif args.ood_task == 'far':
        print(f"Far OOD Detection for {args.dataset}\n")
        for ood in far_ood:
            out_loader = pick_dataset(name = ood, train=False, batch_size=args.batch_size, img_size=img_size, num_workers=args.num_workers, patches=args.patches)
            auroc, fpr95, in_array, _ = model.outlier_detection(in_loader, out_loader, display=False, in_array=in_array, patches=args.patches)
            print(f"OOD: {ood}\nAUROC: {auroc:.4f}\nFPR95: {fpr95:.4f}\n\n")

    elif args.ood_task == 'covar':
        print(f"Covariate Shift Detection for {args.dataset}\n")
        corruptions.sort()
        aurocs = []
        fpr95s = []
        for c in corruptions:
            for i in range(1,6):
                out_loader = imagenetc_dataloader(args.batch_size, c, i, num_workers=args.num_workers, patches=args.patches)
                auroc, fpr95, in_array, _ = model.outlier_detection(in_loader, out_loader, display=False, in_array=in_array)
                print(f"OOD: {c} ({i})\nAUROC: {auroc:.4f}\nFPR95: {fpr95:.4f}\n")
                aurocs.append(auroc)
                fpr95s.append(fpr95)
        print(f"Mean AUROC: {np.mean(aurocs):.4f}\nMean FPR95: {np.mean(fpr95s):.4f}\n")