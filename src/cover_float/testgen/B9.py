# Created By: Ryan Wolk (rwolk@hmc.edu) on 1/29/2026

import random
from collections.abc import Generator
from pathlib import Path
from typing import TextIO
from random import seed

from cover_float.common.constants import EXPONENT_BITS, FLOAT_FMTS, MANTISSA_BITS, OP_DIV, OP_MUL, OP_SQRT
from cover_float.common.util import generate_test_vector
from cover_float.reference import run_and_store_test_vector
from cover_float.common.util import reproducible_hash

"""
From Ahronhi et al 2008:

B9. Special Significands on Inputs
This model tests special patterns in the significands of the input operands. Each
of the input operands should contain one of the following patterns (each
sequence can be of length 0 up to the number of bits in the significand - the
more interesting cases will be chosen).
i. A sequence of leading zeroes
ii. A sequence of leading ones
iii. A sequence of trailing zeroes
iv. A sequence of trailing ones
v. A small number of 1s as compared to 0s
vi. A small number of 0s as compared to 1s
vii. A "checkerboard" pattern (for example 00110011… or 011011011…)
viii. Long sequences of 1s
ix. Long sequences of 0s
Operation: Divide, Remainder, Square-root, Multiply
Enable Bits: XE
"""

TWO_SRC_OPS = [
    OP_DIV,
    # OP_REM,
    OP_MUL,
]

ONE_SRC_OPS = [
    OP_SQRT,
]


def generate_checkerboards(length: int) -> Generator[str, None, None]:
    for zeros_length in range(1, length // 2 + 1):
        # for ones_length in range(1, length // 2 + 1):
        ones_length = zeros_length

        pattern = "0" * zeros_length + "1" * ones_length
        pattern *= length // (zeros_length + ones_length) + 1
        yield pattern[:length]
        yield pattern[::-1][:length]


def generate_leading_and_trailing_zeros(length: int) -> Generator[str, None, None]:
    for zeros_length in range(1, length + 1):
        pattern = "0" * zeros_length + "1" + bin(random.getrandbits(length))[2:]
        yield pattern[:length]
        yield pattern[:length][::-1]


def generate_leading_and_trailing_ones(length: int) -> Generator[str, None, None]:
    for ones_length in range(1, length + 1):
        pattern = "1" * ones_length + "0" + bin(random.getrandbits(length))[2:]
        yield pattern[:length]
        yield pattern[:length][::-1]


def generate_with_k_ones(k: int, length: int, limit: int) -> Generator[str, None, None]:
    for _ in range(limit):
        pattern = random.sample(range(length), k)
        bits = ["0"] * length
        for i in pattern:
            bits[i] = "1"

        yield "".join(bits)


def generate_with_long_runs(min_run_length: int, length: int) -> Generator[str, None, None]:
    # for start in range(length - min_run_length + 1):
    start = random.randint(0, length - min_run_length)

    # Generate a pattern with random digits
    pattern = list(bin(random.getrandbits(length))[2:].zfill(length))

    # Fill in a run of ones
    ones_pattern = pattern[:]
    for i in range(start, start + min_run_length):
        ones_pattern[i] = "1"
    yield "".join(ones_pattern)

    zeros_pattern = pattern[:]
    for i in range(start, start + min_run_length):
        zeros_pattern[i] = "0"
    yield "".join(zeros_pattern)


def generate_special_significands(fmt: str) -> list[str]:
    mantissa_length = MANTISSA_BITS[fmt]
    ans: list[str] = []

    # Leading zeros
    ans.extend(generate_leading_and_trailing_zeros(mantissa_length))

    # Leading ones
    ans.extend(generate_leading_and_trailing_ones(mantissa_length))

    # Small number of 1s
    for k in range(1, mantissa_length // 2 + 1):
        ans.extend(generate_with_k_ones(k, mantissa_length, 1))

    # Small number of 0s
    for k in range(1, mantissa_length // 2 + 1):
        ans.extend(generate_with_k_ones(mantissa_length - k, mantissa_length, 1))

    # Checkerboard patterns
    ans.extend(generate_checkerboards(mantissa_length))

    # Long runs of 1s and 0s
    for run_length in range(1, mantissa_length + 1):
        ans.extend(generate_with_long_runs(run_length, mantissa_length))

    return ans


def generate_test_vectors(special_significands: list[str], fmt: str, test_f: TextIO, cover_f: TextIO) -> int:
    total_tests = len(special_significands) ** 2 * len(TWO_SRC_OPS) + len(special_significands) * len(ONE_SRC_OPS)

    for significand_a in special_significands:
        sign = "0"  # We need positive for sqrt

        exponent = bin(2 ** (EXPONENT_BITS[fmt] - 1))[2:].zfill(EXPONENT_BITS[fmt])
        a = sign + exponent + significand_a
        a = int(a, 2)

        # The only SRC1_OP is SQRT
        test_vector = generate_test_vector(OP_SQRT, a, 0, 0, fmt, fmt)
        run_and_store_test_vector(test_vector, test_f, cover_f)

        for significand_b in special_significands:
            for op in TWO_SRC_OPS:
                sign = "0"
                exponent = bin(2 ** (EXPONENT_BITS[fmt] - 1))[2:].zfill(EXPONENT_BITS[fmt])
                a = sign + exponent + significand_a
                # breakpoint()
                a = int(a, 2)

                sign = "0"  # Only positive
                # The idea here is to get a biased exponent of 1
                exponent = bin(2 ** (EXPONENT_BITS[fmt] - 1))[2:].zfill(EXPONENT_BITS[fmt])
                b = sign + exponent + significand_b
                b = int(b, 2)

                test_vector = generate_test_vector(op, a, b, 0, fmt, fmt)

                run_and_store_test_vector(test_vector, test_f, cover_f)

    return total_tests


def main() -> None:
    total_tests: int = 0

    with (
        Path("tests/testvectors/B9_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B9_cv.txt").open("w") as cover_f,
    ):
        for fmt in FLOAT_FMTS:
            hashval = reproducible_hash(fmt + "b9")
            seed(hashval)
            special_significands = generate_special_significands(fmt)

            generated = generate_test_vectors(special_significands, fmt, test_f, cover_f)
            total_tests += generated

    print(f"Generated {total_tests} tests for B9.")


if __name__ == "__main__":
    main()
