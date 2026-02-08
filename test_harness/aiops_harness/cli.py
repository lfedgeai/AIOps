import argparse
import os
from rich.console import Console
from .runner import run_experiment, run_benchmark


console = Console()


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="aiops-harness", description="AIOps LLM Test Harness")
	sub = parser.add_subparsers(dest="command", required=True)

	run_p = sub.add_parser("run", help="Run a single experiment")
	run_p.add_argument("--config", required=True, help="Path to YAML config")
	run_p.add_argument("--dataset", required=True, help="Path to dataset directory")
	run_p.add_argument("--seed", type=int, default=42)

	bench_p = sub.add_parser("benchmark", help="Run multiple seeds/configs")
	bench_p.add_argument("--config", required=True, help="Path to YAML config")
	bench_p.add_argument("--dataset", required=True, help="Path to dataset directory")
	bench_p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if args.command == "run":
		console.log(f"Running experiment: config={args.config} dataset={args.dataset} seed={args.seed}")
		run_experiment(args.config, args.dataset, seed=args.seed)
	elif args.command == "benchmark":
		console.log(f"Running benchmark: config={args.config} dataset={args.dataset} seeds={args.seeds}")
		run_benchmark(args.config, args.dataset, seeds=args.seeds)
	else:
		parser.print_help()

