"""Earlier local-search CLI and backward-compatible import path.

The active competitive solver is ``python -m solver.competitive``.
The local-search implementation is now ``solver.heuristics.local_search``.
"""
import argparse
from pathlib import Path
from solver.paths import ROOT
from solver.data.sparse import SparseNetwork
from solver.model.chromosome import to_paths, to_chromosome
from solver.evaluation.fast import Score, FastEvaluator
from solver.heuristics.local_search import Optimizer


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'data/database.npz')
    ap.add_argument('--resume',type=Path,default=None,help='Start from existing verified .npy')
    ap.add_argument('--minutes',type=float,default=3)
    ap.add_argument('--iterations',type=int,default=None)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=ap.parse_args()
    opt=Optimizer(args.data,args.output_dir,args.seed)
    if args.resume is not None: initial=args.resume
    elif opt.checkpoint.exists():initial=opt.checkpoint
    else:initial=ROOT/'data/example.npy'
    opt.init(initial)
    opt.run(args.minutes,args.iterations)

if __name__=='__main__':main()
