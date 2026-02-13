import argparse
from cover_float.reference import run_test_vector

import cover_float.testgen as tg

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

    with open(args.input_file, "r") as infile, open(args.output_file, "w") as outfile:
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
    args = parser.parse_args()

    if args.models is None:
        tg.B1.main()
        tg.B9.main()
        tg.B10.main()
        tg.B11.main()
        tg.B12.main()
    else:
        if "B1" in args.models:
            tg.B1.main()
        if "B9" in args.models:
            tg.B9.main()
        if "B10" in args.models:
            tg.B10.main()
        if "B11" in args.models:
            tg.B11.main()
        if "B12" in args.models:
            tg.B12.main()
