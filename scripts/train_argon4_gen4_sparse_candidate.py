"""Train isolated phase-two sparse candidates only after frozen registration."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration",required=True)
    parser.add_argument("--threads",type=int,choices=(2,),default=2)
    parser.add_argument("--phase",choices=("both","oos_2026","deployment_q4"),default="both")
    parser.add_argument("--streams",nargs="+")
    parser.add_argument("--device",choices=("cpu","cuda"),default=None)
    args=parser.parse_args()
    from tools.argon4_forward.gen4_sparse_candidate import train_all_sparse
    phases=("oos_2026","deployment_q4") if args.phase=="both" else (args.phase,)
    train_all_sparse(preregistration_path=args.preregistration,threads=args.threads,
                     phases=phases,streams=args.streams,device=args.device)


if __name__=="__main__":
    main()

