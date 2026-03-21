"""
Angela Zheng

March 3, 2026

SUMMARY
This script generates test vectors for the B2 model: Near FP Base Values - Hamming Distance.
It takes specific boundary values (Zero, One, MinSubNorm, etc.) and enumerates over
small deviations by flipping one bit of the significand at a time.

DEFINITION
Base Values: Zero, One, MinSubNorm, MaxSubNorm, MinNorm, MaxNorm
Operations: add, sub, multiply, fmadd, fmsub, fnmadd, fnmsub, sqrt
Total test vectors generated: TBD

For 32 b fp,
Zero        0.000000 P-126
One         1.000000 P0
Minsubnorm  0.000001 P-126
Maxsubnorm  0.7FFFFF P-126
MinNorm     1.000000 P-126 later ones have problem
MaxNorm     1.7FFFFF P127 rounds to infinity
"""

import random
from pathlib import Path
from typing import TextIO

from cover_float.common.constants import (
    EXPONENT_BITS,
    FLOAT_FMTS,
    MANTISSA_BITS,
    OP_ADD,
    OP_DIV,
    OP_FMADD,
    OP_FMSUB,
    OP_FNMADD,
    OP_FNMSUB,
    OP_MUL,
    OP_SQRT,
    OP_SUB,
    ROUND_NEAR_EVEN,
)
from cover_float.reference import run_and_store_test_vector, run_test_vector


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    """Converts binary fp components into a 32-character padded hex string."""
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"


def get_result_from_ref(op: str, a: str, b: str, c: str, fmt: str) -> str:
    """Calls reference model and extracts the result hex."""
    vector = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00"
    res_str = run_test_vector(vector)
    return res_str.split("_")[6]


# steps is only going to be 1 or -1. So this function add or subtract one to the original number
def calibrate(hex_val: str, steps: int) -> str:
    """Adds or subtracts from the integer representation to step by ULPs."""
    val_int = int(hex_val, 16)
    return f"{(val_int + steps):032X}"


# TODO: looks like the last few minnorm results are a bit problematic.
# TODO: further constraint the exponent range to make it possible for random generation of operand exponents
# Workflow: take 23 bits for single precision, tthen randomly generates a exponents.
# then use softfloat to get b by reversing the operation, and then add or subtract 1 from b until the result is as
# expected.
def test_add(fmt: str, desired_result: str, base_e: int, sign: int, test_f: TextIO, cover_f: TextIO) -> None:
    # We constrain the exponent of a to not be too far away from the base, so that there is enough precision
    # to produce a b different from a that would result in a very small value intead of 0.
    m_bits = MANTISSA_BITS[fmt]

    a_exp = base_e  # set a_exp to base_e because since when adding and subtracting numbers with different exponents,
    # the exponents shift, and the loss of precision cannot be fully compensated with just adding or subtracting 1.

    # for maxnorm, when the desired result is positive, the P127 operand must be positive, because if it is negative
    # the other operand will need to be bigger than the max allowed number, which will round to positive infinity
    # when the desired result is negative, the P127 operand must be negative for the same reason.
    if base_e == ((1 << EXPONENT_BITS[fmt]) - 2):
        if sign == 0:
            a = decimalComponentsToHex(fmt, 0, a_exp, random.getrandbits(m_bits))
            b = get_result_from_ref(OP_SUB, desired_result, a, "0" * 32, fmt)
            run_and_store_test_vector(
                f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
            )
        else:
            a = decimalComponentsToHex(fmt, 1, a_exp, random.getrandbits(m_bits))
            b = get_result_from_ref(OP_SUB, desired_result, a, "0" * 32, fmt)
            run_and_store_test_vector(
                f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
            )
        return

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(m_bits))
    b = get_result_from_ref(OP_SUB, desired_result, a, "0" * 32, fmt)

    run_and_store_test_vector(
        f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


# TODO: looks like a few minnorm results are a bit problematic.
def test_sub(fmt: str, desired_result: str, base_e: int, sign: int, test_f: TextIO, cover_f: TextIO) -> None:
    m_bits = MANTISSA_BITS[fmt]

    a_exp = base_e
    if base_e == ((1 << EXPONENT_BITS[fmt]) - 2):
        if sign == 0:
            a = decimalComponentsToHex(fmt, 0, a_exp, random.getrandbits(m_bits))
            b = get_result_from_ref(OP_SUB, a, desired_result, "0" * 32, fmt)
            run_and_store_test_vector(
                f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
            )
        else:
            a = decimalComponentsToHex(fmt, 1, a_exp, random.getrandbits(m_bits))
            b = get_result_from_ref(OP_SUB, a, desired_result, "0" * 32, fmt)
            run_and_store_test_vector(
                f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
            )
        return
    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = get_result_from_ref(OP_SUB, a, desired_result, "0" * 32, fmt)

    run_and_store_test_vector(
        f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_mul(fmt: str, desired_result: str, base_e: int, test_f: TextIO, cover_f: TextIO) -> None:
    # Also has to restrain exponent so b doesn't underflow. Ex. we want exp -126, if a_exp is 32, we need b_exp -158
    # which clips to -126.
    max_exp = (1 << EXPONENT_BITS[fmt]) - 2
    maxnorm_exp = (1 << EXPONENT_BITS[fmt]) - 2
    bias = 2 ** (EXPONENT_BITS[fmt] - 1) - 1
    if base_e != maxnorm_exp:
        min_safe_exp = 0
        max_safe_exp = bias
    else:
        min_safe_exp = bias
        max_safe_exp = max_exp

    a_exp = random.randint(min_safe_exp, max_safe_exp)
    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))

    b = get_result_from_ref(OP_DIV, desired_result, a, "0" * 32, fmt)

    ans = get_result_from_ref(OP_MUL, a, b, "0" * 32, fmt)
    if ans > desired_result:
        b = calibrate(b, -1)
    elif ans < desired_result:
        b = calibrate(b, 1)

    run_and_store_test_vector(
        f"{OP_MUL}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_div(fmt: str, desired_result: str, base_e: int, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = (1 << EXPONENT_BITS[fmt]) - 2
    b_exp = random.randint(0, max_exp)
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))

    a = get_result_from_ref(OP_MUL, desired_result, b, "0" * 32, fmt)

    ans = get_result_from_ref(OP_DIV, a, b, "0" * 32, fmt)
    if ans < desired_result:
        a = calibrate(a, 1)
    elif ans > desired_result:
        a = calibrate(a, -1)

    run_and_store_test_vector(
        f"{OP_DIV}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_sqrt(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    a = get_result_from_ref(OP_MUL, desired_result, desired_result, "0" * 32, fmt)

    if get_result_from_ref(OP_SQRT, a, "0" * 32, "0" * 32, fmt) != desired_result:
        if get_result_from_ref(OP_SQRT, calibrate(a, 1), "0" * 32, "0" * 32, fmt) == desired_result:
            a = calibrate(a, 1)
        elif get_result_from_ref(OP_SQRT, calibrate(a, -1), "0" * 32, "0" * 32, fmt) == desired_result:
            a = calibrate(a, -1)

    run_and_store_test_vector(
        f"{OP_SQRT}_{ROUND_NEAR_EVEN}_{a}_{32 * '0'}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_fmadd(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = (1 << EXPONENT_BITS[fmt]) - 2
    a = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )
    b = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )

    c = get_result_from_ref(OP_FNMADD, a, b, desired_result, fmt)

    if get_result_from_ref(OP_FMADD, a, b, c, fmt) != desired_result:
        if get_result_from_ref(OP_FMADD, a, b, calibrate(c, 1), fmt) == desired_result:
            c = calibrate(c, 1)
        elif get_result_from_ref(OP_FMADD, a, b, calibrate(c, -1), fmt) == desired_result:
            c = calibrate(c, -1)

    run_and_store_test_vector(f"{OP_FMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def test_fmsub(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = (1 << EXPONENT_BITS[fmt]) - 2
    a = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )
    b = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )

    c = get_result_from_ref(OP_FMSUB, a, b, desired_result, fmt)

    if get_result_from_ref(OP_FMSUB, a, b, c, fmt) != desired_result:
        if get_result_from_ref(OP_FMSUB, a, b, calibrate(c, 1), fmt) == desired_result:
            c = calibrate(c, 1)
        elif get_result_from_ref(OP_FMSUB, a, b, calibrate(c, -1), fmt) == desired_result:
            c = calibrate(c, -1)

    run_and_store_test_vector(f"{OP_FMSUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def test_fnmadd(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = (1 << EXPONENT_BITS[fmt]) - 2
    a = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )
    b = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )

    c = get_result_from_ref(OP_FMADD, a, b, desired_result, fmt)

    if get_result_from_ref(OP_FNMADD, a, b, c, fmt) != desired_result:
        if get_result_from_ref(OP_FNMADD, a, b, calibrate(c, 1), fmt) == desired_result:
            c = calibrate(c, 1)
        elif get_result_from_ref(OP_FNMADD, a, b, calibrate(c, -1), fmt) == desired_result:
            c = calibrate(c, -1)

    run_and_store_test_vector(f"{OP_FNMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def test_fnmsub(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = (1 << EXPONENT_BITS[fmt]) - 2
    a = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )
    b = decimalComponentsToHex(
        fmt, random.randint(0, 1), random.randint(0, max_exp), random.getrandbits(MANTISSA_BITS[fmt])
    )

    c = get_result_from_ref(OP_FNMSUB, a, b, desired_result, fmt)

    if get_result_from_ref(OP_FNMSUB, a, b, c, fmt) != desired_result:
        if get_result_from_ref(OP_FNMSUB, a, b, calibrate(c, 1), fmt) == desired_result:
            c = calibrate(c, 1)
        elif get_result_from_ref(OP_FNMSUB, a, b, calibrate(c, -1), fmt) == desired_result:
            c = calibrate(c, -1)

    run_and_store_test_vector(f"{OP_FNMSUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def main() -> None:
    with (
        Path("./tests/testvectors/B2_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B2_cv.txt").open("w") as cover_f,
    ):
        for fmt in FLOAT_FMTS:
            m_bits = MANTISSA_BITS[fmt]

            bases = {
                "Zero": (0, 0),
                "One": (0, (1 << (EXPONENT_BITS[fmt] - 1)) - 1),  # ?
                "MinSub": (1, 0),
                "MaxSub": ((1 << m_bits) - 1, 0),
                "MinNorm": (0, 1),
                "MaxNorm": ((1 << m_bits) - 1, (1 << EXPONENT_BITS[fmt]) - 2),
            }

            for _, (base_m, base_e) in bases.items():
                for i in range(m_bits):
                    desired_m = base_m ^ (1 << i)
                    for sign in [0, 1]:
                        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)

                        # test_add(fmt, desired_result, base_e, sign, test_f, cover_f)
                        test_sub(fmt, desired_result, base_e, sign, test_f, cover_f)
                        # test_mul(fmt, desired_result, base_e, test_f, cover_f)
                        # test_div(fmt, desired_result, base_e, test_f, cover_f)

                        # if sign == 0:
                        #     test_sqrt(fmt, desired_result, test_f, cover_f)

                        # test_fmadd(fmt, desired_result, test_f, cover_f)
                        # test_fmsub(fmt, desired_result, test_f, cover_f)
                        # test_fnmadd(fmt, desired_result, test_f, cover_f)
                        # test_fnmsub(fmt, desired_result, test_f, cover_f)


if __name__ == "__main__":
    main()
