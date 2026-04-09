"""
Angela Zheng (angela20061015@gmail.com)
"""

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
    OP_FMADD,
    OP_FMSUB,
    OP_FNMADD,
    OP_FNMSUB,
    ROUND_NEAR_EVEN,
)
from cover_float.common.util import reproducible_hash
from cover_float.reference import run_and_store_test_vector


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    """Converts binary fp components into a 32-character padded hex string."""
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"


# def makeFMATestVectors(fmt: str, d: int, op: str, test_f: TextIO, cover_f: TextIO) -> None:
#     m = MANTISSA_BITS[fmt]
#     p = m + 1
#     bias = BIAS[fmt]
#     min_exp, max_exp = BIASED_EXP[fmt]

#     # Align product exponent with C
#     a_exp = random.randint(bias, max_exp - 2)
#     test = max_exp - a_exp
#     print(test)
#     b_exp = random.randint(0, test)
#     prod_exp = a_exp + b_exp - bias

#     # Determine signs for subtractive interaction
#     a_sign = 0
#     b_sign = 0
#     prod_sign = a_sign ^ b_sign
#     if op in [OP_FMADD, OP_FNMSUB]:
#         c_sign = 1 - prod_sign
#     else:
#         c_sign = prod_sign

#     # d = -(2p + 1) (Maximum Cancellation) ---
#     if d == -(2 * p + 1):
#         a_m, b_m = 0, 0
#         c_m = (1 << m) - 1
#         c_exp = prod_exp - 1

#     # d = 1 (Carry) ---
#     elif d == 1:
#         c_sign = prod_sign if op in [OP_FMADD, OP_FNMSUB] else 1 - prod_sign
#         a_m, b_m = (1 << m) - 1, (1 << m) - 1
#         c_m = (1 << m) - 1
#         c_exp = prod_exp

#     # d = 0 (No Cancellation) ---
#     elif d == 0:
#         a_m, b_m = (1 << m) - 1, 0
#         c_m = ((1 << m) - 1) & ~1
#         c_exp = prod_exp - 1

#     # other cases
#     else:
#         k = -d
#         c_exp = prod_exp
#         a_m = random.getrandbits(m)
#         b_m = 0

#         # Use max(0, ...) to prevent negative shift counts
#         shift_amt = max(0, m - k + 1)
#         prefix = (a_m >> shift_amt) << shift_amt
#         c_m_prefix = prefix

#         # only place bit if it's within the m-bit range
#         bit_pos = m - k
#         if bit_pos >= 0:
#             diff_bit = 1 << bit_pos
#         else:
#             # If k > m, the difference happens in the internal "lower"
#             # product bits. For the m-bit C, we just set a small value.
#             diff_bit = 0

#         # prevent negative shift in getrandbits
#         tail_len = max(0, m - k - 2)
#         if tail_len > 0:
#             a_tail = (1 << (tail_len)) | random.getrandbits(tail_len)
#             c_tail = random.getrandbits(tail_len)
#         else:
#             a_tail = 0
#             c_tail = 0

#         mask = (1 << shift_amt) - 1
#         a_m = (a_m & ~mask) | diff_bit | a_tail
#         c_m = c_m_prefix | c_tail

#     a = decimalComponentsToHex(fmt, a_sign, a_exp, a_m)
#     b = decimalComponentsToHex(fmt, b_sign, b_exp, b_m)
#     c = decimalComponentsToHex(fmt, c_sign, c_exp, c_m)

#     run_and_store_test_vector(
#         f"{OP_FMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
#     )
#     run_and_store_test_vector(
#         f"{OP_FMSUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
#     )
#     run_and_store_test_vector(
#         f"{OP_FNMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
#     )
#     run_and_store_test_vector(
#         f"{OP_FNMSUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
#     )


def makeFMATestVectors(fmt: str, d: int, op: str, test_f: TextIO, cover_f: TextIO) -> None:
    m = MANTISSA_BITS[fmt]
    p = m + 1
    bias = BIAS[fmt]
    min_exp, max_exp = BIASED_EXP[fmt]

    a_exp = random.randint(min_exp, max_exp - 2)
    b_exp = random.randint(min_exp, max_exp - 2)

    a_m = random.getrandbits(m)
    b_m = random.getrandbits(m)

    # Approximate product exponent
    prod_exp = a_exp + b_exp - bias

    prod_m = random.getrandbits(m)

    a_sign = random.getrandbits(1)
    b_sign = random.getrandbits(1)
    prod_sign = a_sign ^ b_sign

    c_sign = 1 - prod_sign if op in [OP_FMADD, OP_FNMSUB] else prod_sign

    # control cancellation depth:
    k = -d  # number of bits that cancel

    c_exp = prod_exp

    if d == 1:
        # Carry case
        c_sign = prod_sign
        c_m = (1 << m) - 1

    elif d == -(2 * p + 1):
        # Maximum cancellation → almost equal
        c_exp = prod_exp - 1
        c_m = prod_m ^ 1  # tiny diff

    else:
        # General case
        if k <= m:
            # Match top k bits
            prefix_mask = ((1 << k) - 1) << (m - k)
            prefix = prod_m & prefix_mask

            # Flip next bit
            flip_pos = m - k - 1
            flip_bit = 1 << flip_pos if flip_pos >= 0 else 0

            # Random tail
            tail_len = max(0, flip_pos)
            tail = random.getrandbits(tail_len) if tail_len > 0 else 0

            c_m = prefix | flip_bit | tail

        else:
            # Deep cancellation beyond mantissa
            c_m = prod_m
            c_exp = prod_exp - (k - m)

    a = decimalComponentsToHex(fmt, a_sign, a_exp, a_m)
    b = decimalComponentsToHex(fmt, b_sign, b_exp, b_m)
    c = decimalComponentsToHex(fmt, c_sign, c_exp, c_m)

    for opcode in [OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]:
        run_and_store_test_vector(
            f"{opcode}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00",
            test_f,
            cover_f,
        )


def main() -> None:
    with (
        Path("./tests/testvectors/B16_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B16_cv.txt").open("w") as cover_f,
    ):
        for fmt in FLOAT_FMTS:
            p = MANTISSA_BITS[fmt] + 1
            # Range adjusted for FMA: [-(2*p + 1), 1]
            for d in range(-(2 * p + 1), 2):
                for op in [OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]:
                    seed(reproducible_hash(f"{fmt}_b16_{d}_{op}"))
                    makeFMATestVectors(fmt, d, op, test_f, cover_f)


if __name__ == "__main__":
    main()
