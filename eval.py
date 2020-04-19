import torch
from models import TSPAgent
from utils import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from problems import TSP
from argparse import ArgumentParser


def main(args):
    model = TSPAgent.load_from_checkpoint(args.ckpt_path).net.eval()
    if torch.cuda.is_available() and args.gpu:
        model = model.cuda()
    test_set = load_dataset(args.test_data)
    print(f'Loaded pretrained model and test data set.')
    data_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    rollout = False if args.n_sampling > 1 else True
    costs = []
    with torch.no_grad():
        for batch in tqdm(iter(data_loader), total=len(data_loader)):
            if torch.cuda.is_available() and args.gpu:
                batch = batch.cuda()
            cost = []
            for _ in range(args.n_sampling):
                perm, _ = model(batch, rollout)
                c = TSP(batch, perm).cpu()
                cost += [c]
            cost = torch.cat(cost, 1).min(1)[0]
            costs += [cost]
        costs = torch.cat(costs, 0)
        avg_cost = cost.mean().item()

        print(f'Pretrained model: {args.ckpt_path}')
        print(f'Test set:         {args.test_data}')
        print(f'Avg cost:         {avg_cost}')


if __name__ == "__main__":
    parser = ArgumentParser()

    # check point
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='pretrained/tsp20_pretrained.ckpt')

    # test set
    parser.add_argument('--test_data',
                        type=str,
                        default='data/tsp/tsp20_test_seed1234.pkl')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)

    # sampling or greedy
    parser.add_argument('--n_sampling', type=int, default=1)

    # use gpu or not
    parser.add_argument('--gpu', type=bool, default=True)

    args = parser.parse_args()
    main(args)
