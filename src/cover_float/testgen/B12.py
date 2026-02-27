"""
Angela Zheng

February 10, 2026

SUMMARY
This script generates cancellation test vectors for the B12 model:

    For the difference d between the exponent of the intermediate result
    and the maximum exponent of the inputs:
        d ∈ [-p, +1]
    Enable Bits: XE

DEFINITION
p: precision of the format, including the hidden 1. (# mantissa bits + 1)
a: first operand
b: second operand
d: the difference between max(a_exp, b_exp) and exponent of the intermediate result

Total test vectors generated: 438
"""
# TODO: For future: implement logic to get different a and b exponents in regular cases

import random
from random import seed
from pathlib import Path
from typing import TextIO

from cover_float.common.constants import (
    BIASED_EXP,
    EXPONENT_BITS,
    FLOAT_FMTS,
    MANTISSA_BITS,
    OP_ADD,
    OP_SUB,
    ROUND_NEAR_EVEN,
)
from cover_float.reference import run_and_store_test_vector
from cover_float.common.util import reproducible_hash

vector_count = 0


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"


def write_add(fmt: str, a_hex: str, b_hex: str, test_f: TextIO, cover_f: TextIO) -> None:
    run_and_store_test_vector(
        f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00\n", test_f, cover_f
    )


def write_sub(fmt: str, a_hex: str, b_hex: str, test_f: TextIO, cover_f: TextIO) -> None:
    run_and_store_test_vector(
        f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00\n", test_f, cover_f
    )


def makeCancellationMantissas(fmt: str, d: int) -> tuple[int, int]:
    """
    Generate identical -d bits for both operands such that exactly -d bits cancel.
    """

    # d = -8, k = 8
    # a = 01011011 11 1110111111001    8 random bits + 11 + m-k-2 random bits
    # b = 01011011 00 1110111111001    same 8 random bits + 00 + same m-k-2 random bits
    # The 11 and 00 are to prevent borrowing from the previous bit canceling the differing bit

    m = MANTISSA_BITS[fmt]
    k = -d

    # prefix for a and b
    if d == 0:
        tail = random.getrandbits(m - 2)
        a_m = 1 << (m - 1) | 1 << (m - 2) | tail
        b_m = 0 << (m - 1) | 0 << (m - 2) | tail
        return a_m, b_m
    else:
        a_prefix = 1 << (m - 1) | random.getrandbits(k - 1) << (m - k)
        b_prefix = a_prefix

    # differing bit
    diff_bit = 1 << (m - k - 1)

    # tails
    if k < (m - 1):
        a_tail = 1 << (m - k - 2) | random.getrandbits(m - k - 2)
        b_tail = random.getrandbits(m - k - 2)
    else:
        a_tail = 0
        b_tail = 0

    a_m = a_prefix | diff_bit | a_tail
    b_m = b_prefix | b_tail

    return a_m, b_m


def makeExactCancelMantissas(fmt: str) -> tuple[int, int]:
    """
    Generate mantissas so that exactly m bits cancel.
    """

    # a = 11011011011110111111001 0     0 + 21 random bits (identical) + ends in 1
    # b = 101101101111011111100 01      same 21 bits (identical) + ends in 01
    # b = 11011011011110111111000 1     after b shifts right, it is in alignment to cancel m bits

    m = MANTISSA_BITS[fmt]

    identical = random.getrandbits(m - 2)
    a_m = 1 << (m - 1) | identical << 1 | 1
    b_m = identical | 0 << 1 | 1

    return a_m, b_m


def makeCarryMantissas(fmt: str) -> tuple[int, int]:
    """
    Force carry for d = +1
    """

    m = MANTISSA_BITS[fmt]

    a_m = (1 << m) - 1  # 1.111...111
    b_m = 1 << (m - 1) | 1  # 1.000...001 (LSB set)

    return a_m, b_m


def makeNegPMantissas(fmt: str) -> tuple[int, int]:
    """
    Shifts b exp down 1 to create cancellation of p bits.
    """
    m = MANTISSA_BITS[fmt]

    a_m = 0  # A = 1.00...0 (Mantissa 0)
    b_m = (1 << m) - 1  # B = 1.11...1 (Mantissa all 1s), b_exp = a_exp - 1

    return a_m, b_m


def makeTestVectors(fmt: str, d: int, operation: str, test_f: TextIO, cover_f: TextIO) -> None:
    m = MANTISSA_BITS[fmt]
    p = m + 1
    min_exp, max_exp = BIASED_EXP[fmt]

    is_carry = False
    is_add = operation == "add"
    write_fn = write_add if is_add else write_sub

    # Exponents
    a_exp = random.randint(min_exp - d, max_exp)
    b_exp = a_exp

    # Mantissas
    if d == 1:
        is_carry = True
        a_m, b_m = makeCarryMantissas(fmt)
    elif d == -p:
        a_m, b_m = makeNegPMantissas(fmt)
        b_exp -= 1
    elif d == -m:
        a_m, b_m = makeExactCancelMantissas(fmt)
        b_exp -= 1
    else:
        a_m, b_m = makeCancellationMantissas(fmt, d)

    # Signs
    if is_add:
        if is_carry:
            a_sign = 0
            b_sign = 0
        else:
            a_sign = 0
            b_sign = 1
    else:
        if is_carry:
            a_sign = 0
            b_sign = 1
        else:
            a_sign = 0
            b_sign = 0

    a_hex = decimalComponentsToHex(fmt, a_sign, a_exp, a_m)
    b_hex = decimalComponentsToHex(fmt, b_sign, b_exp, b_m)

    write_fn(fmt, a_hex, b_hex, test_f, cover_f)


def CancellationTests(test_f: TextIO, cover_f: TextIO, fmt: str) -> None:
    p = MANTISSA_BITS[fmt] + 1

    for d in range(-p, 2):  # [-p, +1]
        hashval = reproducible_hash(OP_ADD + fmt + "b12")
        seed(hashval)
        makeTestVectors(fmt, d, "add", test_f, cover_f)
        hashval = reproducible_hash(OP_SUB + fmt + "b12")
        seed(hashval)
        makeTestVectors(fmt, d, "sub", test_f, cover_f)


def main() -> None:
    with (
        Path("./tests/testvectors/B12_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B12_cv.txt").open("w") as cover_f,
    ):
        test_f.write("// Cancellation tests\n")
        test_f.write("// Operations: ADD, SUB\n")

        for fmt in FLOAT_FMTS:
            CancellationTests(test_f, cover_f, fmt)


if __name__ == "__main__":
    main()
