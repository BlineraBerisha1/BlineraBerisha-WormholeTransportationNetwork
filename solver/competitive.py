"""Active ESA SpOC2 optimizer CLI and backward-compatible imports.

The working implementation is divided into solver.data, solver.evaluation,
solver.model, solver.heuristics and solver.search.
Run ``python -m solver.competitive --minutes 5`` to search.
"""
import argparse
from pathlib import Path
from solver.paths import ROOT
from solver.data.csr import CSRNetwork
from solver.heuristics.route_builder import _build_one, _node_potentials
from solver.search.alns import CompetitiveSolver


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/database.npz')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--resume',type=Path,default=None)
    parser.add_argument('--minutes',type=float,default=10.)
    parser.add_argument('--seed',type=int,default=20261115)
    parser.add_argument('--construction-trials',type=int,default=40)
    parser.add_argument('--windows',type=int,default=12)
    parser.add_argument('--alns-trials',type=int,default=30)
    parser.add_argument('--discount',type=float,default=.80,help='Discount factor for Lagrangian graph lookahead, 0..1')
    args=parser.parse_args()
    opt=CompetitiveSolver(args.data,args.output_dir,args.seed)
    opt.discount=args.discount
    opt.init([args.resume] if args.resume is not None else [])
    opt.run(args.minutes,args.construction_trials,args.windows,args.alns_trials)

if __name__=='__main__':main()
