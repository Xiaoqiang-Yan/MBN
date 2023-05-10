import argparse

parser = argparse.ArgumentParser(description='MBN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default="pubmed")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_z', type=int, default=10)
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta', type=int, default=10)

args = parser.parse_args()
print("Network setting…")

if args.name == 'acm':
    args.k = None
    args.lr = 4e-5
    args.n_clusters = 3
    args.n_input = 1870
    args.alpha = 0.03
    args.beta = 0.04
elif args.name == 'dblp':
    args.k = None
    args.lr = 2e-3
    args.n_clusters = 4
    args.n_input = 334
    args.alpha = 0.05
    args.beta = 0.1
elif args.name == 'cite':
    args.k = None
    args.lr = 4e-5
    args.n_clusters = 6
    args.n_input = 3703
    args.alpha = 0.03
    args.beta = 0.03
elif args.name == 'cora':
    args.k = None
    # args.lr = 1e-4
    args.lr = 9e-5
    args.n_clusters = 7
    args.n_input = 1433
    # args.alpha = 0.1
    # args.beta = 0.6
    args.alpha = 10
    args.beta = 50
elif args.name == 'pubmed':
    args.k = None
    args.lr = 4e-4
    args.n_clusters = 3
    args.n_input = 500
    args.alpha = 0.08
    args.beta = 0.27
else:
    print("error!")
