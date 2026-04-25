import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cover_float.common.log as log
import cover_float.testgen as tg
from cover_float.common.constants import config
from cover_float.common.util import SingleThreadedExecutor
from cover_float.reference import run_test_vector

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input test vector file")
    parser.add_argument("output_file", type=str, help="Path to the output cover vector file")
    parser.add_argument(
        "--suppress-error-check",
        action="store_true",
        help="Suppress error checking between expected and actual results",
    )
    args = parser.parse_args()

    with Path(args.input_file).open("r") as infile, Path(args.output_file).open("w") as outfile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith("//"):
                continue  # Skip empty lines and comments
            result = run_test_vector(line, args.suppress_error_check)
            outfile.write(result)


def testgen() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        action="extend",
        nargs="+",
        help="Model(s) to generate test vectors for",
    )
    parser.add_argument("--output-dir", type=str, default="tests", help="Directory to save generated test vectors")
    parser.add_argument("--single-thread", action="store_true", help="Run Generation in a Single Thread")
    parser.add_argument("--jobs", type=int, default=None, help="Number of Jobs to Run When Multi-Threaded")
    parser.add_argument(
        "--partial-output", action="store_true", help="Create a Reduced Number of Tests in Test Heavy Models"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    single_thread = args.single_thread or (args.models is not None and len(args.models) < 2)
    config.FULL_COVERAGE_TESTGEN = not args.partial_output

    if single_thread:
        executor = SingleThreadedExecutor()
    else:
        executor = ProcessPoolExecutor() if args.jobs is None else ProcessPoolExecutor(max_workers=args.jobs)

    with log.StatusReporter() as logger, executor:
        if args.models is None:
            for model in tg.model.GLOBAL_MODELS:
                tg.model.GLOBAL_MODELS[model](output_dir, logger, executor)
        else:
            for model in args.models:
                if model in tg.model.GLOBAL_MODELS:
                    tg.model.GLOBAL_MODELS[model](output_dir, logger, executor)
