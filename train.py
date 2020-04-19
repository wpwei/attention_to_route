from pytorch_lightning import Trainer
from models import TSPAgent
from argparse import ArgumentParser


def main(args):
    model = TSPAgent(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.save_checkpoint(f'tsp{args.n_node}_ep{trainer.current_epoch}.ckpt')


if __name__ == '__main__':
    parser = ArgumentParser()

    # hparams
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--gradient_clip_val', type=int, default=1)

    # network structure
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--k_dim', type=int, default=16)
    parser.add_argument('--v_dim', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)

    # train set
    parser.add_argument('--n_batch_per_epoch', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_node', type=int, default=20)

    # validation set
    parser.add_argument('--val_set', type=str, default=None)

    # baseline set
    parser.add_argument('--baseline_set_size', type=int, default=10000)

    # num_workers of DataLoaders
    parser.add_argument('--num_workers', type=int, default=4)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
