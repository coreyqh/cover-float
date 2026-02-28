import argparse
from pathlib import Path

import cover_float.testgen as tg
from cover_float.reference import run_test_vector
from cover_float.scripts.parse_testvectors import format_output, parse_test_vector


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


def auto_parse(model_name: str, output_dir: str) -> None:
    input_path = Path(output_dir) / "testvectors" / f"{model_name}_tv.txt"
    output_path = Path(output_dir) / "readable" / f"{model_name}_parsed.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r") as infile, output_path.open("w") as outfile:
        for line in infile:
            parsed = parse_test_vector(line)
            if parsed:
                outfile.write(format_output(parsed) + "\n")
                count += 1
    print(f"Parsed {count} {model_name} vectors to {output_path}")


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
        auto_parse("B1", args.output_dir)
        tg.B9.main()
        auto_parse("B9", args.output_dir)
        tg.B10.main()
        auto_parse("B10", args.output_dir)
        tg.B12.main()
        auto_parse("B12", args.output_dir)
        tg.B14.main()
    else:
        if "B1" in args.models:
            tg.B1.main()
            auto_parse("B1", args.output_dir)
        if "B9" in args.models:
            tg.B9.main()
            auto_parse("B9", args.output_dir)
        if "B10" in args.models:
            tg.B10.main()
            auto_parse("B10", args.output_dir)
        if "B12" in args.models:
            tg.B12.main()
            auto_parse("B12", args.output_dir)
        if "B14" in args.models:
            tg.B14.main()
