import argparse

parser = argparse.ArgumentParser(description='MBN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default="phy")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_z', type=int, default=10)
parser.add_argument('--alpha', type=int, default=0.1)
parser.add_argument('--beta', type=int, default=0.1)

args = parser.parse_args()
print("Network setting…")

if args.name == 'acm':
    args.lr = 5e-5
    args.n_clusters = 3
    args.n_input = 1870
elif args.name == 'dblp':
    args.lr = 2e-3
    args.n_clusters = 4
    args.n_input = 334
elif args.name == 'cite':
    args.lr = 4e-5
    args.n_clusters = 6
    args.n_input = 3703
elif args.name == 'phy':
    args.lr = 1e-4
    args.n_clusters = 5
    args.n_input = 8415
    args.cuda = False
else:
    print("error!")
