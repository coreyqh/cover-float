"""
Angela Zheng (angela20061015@gmail.com)

Created:        March 24, 2026
Last Edited:    March 24, 2026
"""

# TODO: Perhaps look into mpmath? Using that to generate b might help.

import random
from pathlib import Path
from random import seed
from typing import TextIO

from cover_float.common.constants import (
    BIAS,
    BIASED_EXP,
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
from cover_float.common.util import reproducible_hash
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


# TODO: IDEA: restrain the difference between a_exp and b_exp to increase accuracy!!!! for ALL operations!!!!
# TODO: IDEA: to calibrate, check whether the numerical part of the result and desired result are equal.
# This is because the flag might set the difference.


# TODO: looks like the last few minnorm results are a bit inaccurate.
# TODO: further constraint the exponent range to make it possible for random generation of operand exponents
def test_add(
    fmt: str, desired_result: str, base_e: int, maxnorm: bool, sign: int, test_f: TextIO, cover_f: TextIO
) -> None:
    # We constrain the exponent of a to not be too far away from the base, so that there is enough precision
    # to produce a b different from a that would result in a very small value intead of 0.
    m_bits = MANTISSA_BITS[fmt]

    a_exp = base_e  # set a_exp to base_e because since when adding and subtracting numbers with different exponents,
    # the exponents shift, and the loss of precision cannot be fully compensated with just adding or subtracting 1.

    # for maxnorm, when the desired result is positive, the P127 operand must be positive, because if it is negative
    # the other operand will need to be bigger than the max allowed number, which will round to positive infinity
    # when the desired result is negative, the P127 operand must be negative for the same reason.
    if maxnorm:
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


# TODO: looks like a few minnorm results are a bit inaccurate.
def test_sub(
    fmt: str, desired_result: str, base_e: int, maxnorm: bool, sign: int, test_f: TextIO, cover_f: TextIO
) -> None:
    m_bits = MANTISSA_BITS[fmt]

    a_exp = base_e
    if maxnorm:
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


# TODO: Few cases in minnorm and maxnorm has inaccuracies
def test_mul(fmt: str, desired_result: str, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    # Also has to restrain exponent so b doesn't underflow. Ex. we want exp -126, if a_exp is 32, we need b_exp -158
    # which clips to -126.

    max_exp = BIASED_EXP[fmt][1]  # 254
    bias = BIAS[fmt]  # 127

    # we still check for maxnorm, but sign doesn't matter because we can just make the other operand negative
    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
    else:
        min_safe_exp = 1
        max_safe_exp = bias

    a_exp = random.randint(min_safe_exp, max_safe_exp)
    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = get_result_from_ref(OP_DIV, desired_result, a, "0" * 32, fmt)

    run_and_store_test_vector(
        f"{OP_MUL}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


# TODO: A few cases in maxnorm has inaccuracies
def test_div(fmt: str, desired_result: str, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    # Also has to restrain exponent so b doesn't overflow. Ex. we want exp -126, if a_exp is 32, we need b_exp 158
    # which clips to 127. Ex. f32/ =0 -1.7631FFP-81 +1.7631FFP66 -> -0.000004P-126 (f32), the exponent difference is 147
    # bc on top of 126 we want to reach we also need to shift the 1 to the right by 5*4+1 = 21. So max shift is when we
    # shift m_bits where we want an exp difference of 149, so we want a_exp + 149 less than 127. This means that a_exp
    # needs to range from -126 to -m_bits.
    max_exp = BIASED_EXP[fmt][1]  # 254
    bias = BIAS[fmt]  # 127
    m_bits = MANTISSA_BITS[fmt]

    # we still check for maxnorm, but sign doesn't matter because we can just make the other operand negative
    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
    else:
        min_safe_exp = 1
        max_safe_exp = bias - m_bits  # For the reason above ^

    a_exp = random.randint(min_safe_exp, max_safe_exp)
    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = get_result_from_ref(OP_DIV, a, desired_result, "0" * 32, fmt)

    run_and_store_test_vector(
        f"{OP_DIV}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_sqrt(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    a = get_result_from_ref(OP_MUL, desired_result, desired_result, "0" * 32, fmt)

    run_and_store_test_vector(
        f"{OP_SQRT}_{ROUND_NEAR_EVEN}_{a}_{32 * '0'}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_fmadd(fmt: str, desired_result: str, base_e: int, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    # We have to limit the freedom here. a and b shouldn't be random because a * b can overflow or underflow,
    # or be other values that makes c +/- infinity.
    max_exp = BIASED_EXP[fmt][1]  # 254
    bias = BIAS[fmt]  # 127
    m_bits = MANTISSA_BITS[fmt]

    # we still check for maxnorm, but sign doesn't matter because we can just make the other operand negative
    # Let's randomly generate a_exp, and constraint b_exp so that c_exp is the same as base_e to make all operand
    # visible in the result (so that neither the product nor operand c is just the result), aka. c is neither 0
    # nor just the desired result.
    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)  # base = 127, a_exp = 53, b_exp
    else:
        min_safe_exp = 1  # base = -126 = 1, a_exp = -125 = 2, b_exp needs to be -1 which is 126
        max_safe_exp = bias  # base = -126 (1), a_exp = -53 (74), b_exp needs to be between -126 - (-53) = -73 (54)
        # and -126 - (-53) + 23 = -50
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    # a*b+c = desired result -> c = -(a*b-desired result)
    c = get_result_from_ref(OP_FNMSUB, a, b, desired_result, fmt)

    run_and_store_test_vector(f"{OP_FMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


# TODO: minsubnorm, maxsubnorm, minnorm all have inaccuracies
def test_fmsub(fmt: str, desired_result: str, base_e: int, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = BIASED_EXP[fmt][1]  # 254
    bias = BIAS[fmt]  # 127
    m_bits = MANTISSA_BITS[fmt]
    # a*b - c = desired_result
    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)
    else:
        min_safe_exp = 1
        max_safe_exp = bias
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    c = get_result_from_ref(OP_FMSUB, a, b, desired_result, fmt)

    run_and_store_test_vector(f"{OP_FMSUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def test_fnmadd(fmt: str, desired_result: str, base_e: int, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = BIASED_EXP[fmt][1]  # 254
    bias = BIAS[fmt]  # 127
    m_bits = MANTISSA_BITS[fmt]  # 23
    # -a*b - c = desired_result
    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)
    else:
        min_safe_exp = 1
        max_safe_exp = bias
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    c = get_result_from_ref(OP_FNMADD, a, b, desired_result, fmt)

    run_and_store_test_vector(f"{OP_FNMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def test_fnmsub(fmt: str, desired_result: str, base_e: int, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = BIASED_EXP[fmt][1]  # 254
    bias = BIAS[fmt]  # 127
    m_bits = MANTISSA_BITS[fmt]
    # -a*b + c = desired_result -> c = a*b + desired_result
    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)
    else:
        min_safe_exp = 1
        max_safe_exp = bias
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    c = get_result_from_ref(OP_FMADD, a, b, desired_result, fmt)

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
                "One": (0, (1 << (EXPONENT_BITS[fmt] - 1)) - 1),
                "MinSub": (1, 0),
                "MaxSub": ((1 << m_bits) - 1, 0),
                "MinNorm": (0, 1),
                "MaxNorm": ((1 << m_bits) - 1, (1 << EXPONENT_BITS[fmt]) - 2),
            }

            for base, (base_m, base_e) in bases.items():
                for i in range(m_bits):
                    desired_m = base_m ^ (1 << i)
                    for sign in [0, 1]:
                        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
                        maxnorm = base == "MaxNorm"
                        seed(reproducible_hash(f"{fmt}_b2_add_{base_e}_{i}_{sign}"))
                        test_add(fmt, desired_result, base_e, maxnorm, sign, test_f, cover_f)
                        seed(reproducible_hash(f"{fmt}_b2_sub_{base_e}_{i}_{sign}"))
                        test_sub(fmt, desired_result, base_e, maxnorm, sign, test_f, cover_f)
                        seed(reproducible_hash(f"{fmt}_b2_mul_{base_e}_{i}_{sign}"))
                        test_mul(fmt, desired_result, maxnorm, test_f, cover_f)
                        seed(reproducible_hash(f"{fmt}_b2_div_{base_e}_{i}_{sign}"))
                        test_div(fmt, desired_result, maxnorm, test_f, cover_f)

                        if sign == 0 and base == "One":
                            seed(reproducible_hash(f"{fmt}_b2_sqrt_{base_e}_{i}_{sign}"))
                            test_sqrt(fmt, desired_result, test_f, cover_f)

                        seed(reproducible_hash(f"{fmt}_b2_fmadd_{base_e}_{i}_{sign}"))
                        test_fmadd(fmt, desired_result, base_e, maxnorm, test_f, cover_f)
                        seed(reproducible_hash(f"{fmt}_b2_fmsub_{base_e}_{i}_{sign}"))
                        test_fmsub(fmt, desired_result, base_e, maxnorm, test_f, cover_f)
                        seed(reproducible_hash(f"{fmt}_b2_fnmadd_{base_e}_{i}_{sign}"))
                        test_fnmadd(fmt, desired_result, base_e, maxnorm, test_f, cover_f)
                        seed(reproducible_hash(f"{fmt}_b2_fnmsub_{base_e}_{i}_{sign}"))
                        test_fnmsub(fmt, desired_result, base_e, maxnorm, test_f, cover_f)


if __name__ == "__main__":
    main()
