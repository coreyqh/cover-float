# B4 Model: Overflow and Near Overflow
#
# This model creates a test-case for each of the following constraints on the
# intermediate results:
#
#   i.   All the numbers in the range [+MaxNorm – 3 ulp, +MaxNorm + 3 ulp]
#   ii.  All the numbers in the range [-MaxNorm – 3 ulp, -MaxNorm + 3 ulp]
#   iii. A random number that is larger than +MaxNorm + 3 ulp
#   iv.  A random number that is smaller than -MaxNorm – 3 ulp
#   v.   One number for every exponent in the range
#        [MaxNorm.exp - 3, MaxNorm.exp + 3] for positive and negative numbers
#
# Operation:     All
# Rounding Mode: All
# Enable Bits:   XE, OE (Both On and both Off)
#
# ULP definition used here:
#   ±1 ULP = last bit of mantissa (LSB=1), guard bit = 0, sticky = 0

import random
from pathlib import Path
from typing import TextIO
from random import seed 

import cover_float.common.constants as const
from cover_float.common.util import reproducible_hash
from cover_float.reference import run_and_store_test_vector

ROUNDING_MODES = [
    const.ROUND_NEAR_EVEN,
    const.ROUND_MINMAG,
    const.ROUND_MIN,
    const.ROUND_MAX,
    const.ROUND_NEAR_MAXMAG,
    const.ROUND_ODD
]

SRC1_OPS = [const.OP_SQRT, const.OP_CLASS]

SRC2_OPS = [
    const.OP_ADD,
    const.OP_SUB,
    const.OP_MUL,
    const.OP_DIV,
    const.OP_REM,
    const.OP_FEQ,
    const.OP_FLT,
    const.OP_FLE,
    const.OP_MIN,
    const.OP_MAX,
    const.OP_FSGNJ,
    const.OP_FSGNJN,
    const.OP_FSGNJX,
]

SRC3_OPS = [const.OP_FMADD, const.OP_FMSUB, const.OP_FNMADD, const.OP_FNMSUB]


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    b_sign = f"{sign:01b}"
    b_exponent = f"{biased_exp:0{const.EXPONENT_BITS[fmt]}b}"
    b_mantissa = f"{mantissa:0{const.MANTISSA_BITS[fmt]}b}"
    b_complete = b_sign + b_exponent + b_mantissa
    h_complete = f"{int(b_complete, 2):032X}"
    return h_complete


def write_exp_range_tests(fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    target_exp = const.BIASED_EXP[fmt][1]  # MaxNorm exponent
    for exp_diff in range(-3, 4):  # Sweep from -3 to +3 ULPs
        biased_exp = target_exp + exp_diff
        for sign in (0, 1):
            mant = random.getrandbits(const.MANTISSA_BITS[fmt])          # integer, like B14
            hex_a = decimalComponentsToHex(fmt, sign, biased_exp, mant)

            for rm in ROUNDING_MODES:
                for op in SRC1_OPS: 
                    hashval = reproducible_hash(op + fmt + "b4")
                    seed(hashval) 
                    run_and_store_test_vector(
                        f"{op}_{rm}_{hex_a}_{32 * '0'}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f, cover_f
                    )
                for op in SRC2_OPS:
                    hashval = reproducible_hash(op + fmt + "b4")
                    seed(hashval) 
                    mant_b = random.getrandbits(const.MANTISSA_BITS[fmt])
                    hex_b = decimalComponentsToHex(fmt, sign, biased_exp, mant_b)
                    run_and_store_test_vector(
                        f"{op}_{rm}_{hex_a}_{hex_b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f, cover_f
                    )
                for op in SRC3_OPS:
                    hashval = reproducible_hash(op + fmt + "b4")
                    seed(hashval) 
                    mant_b = random.getrandbits(const.MANTISSA_BITS[fmt])
                    mant_c = random.getrandbits(const.MANTISSA_BITS[fmt])
                    hex_b = decimalComponentsToHex(fmt, sign, biased_exp, mant_b)
                    hex_c = decimalComponentsToHex(fmt, sign, biased_exp, mant_c)
                    run_and_store_test_vector(
                        f"{op}_{rm}_{hex_a}_{hex_b}_{hex_c}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f, cover_f
                    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def main() -> None:
    with (
        Path("tests/testvectors/B4_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B4_cv.txt").open("w") as cover_f,
    ):
        for fmt in const.FLOAT_FMTS:
            write_exp_range_tests(fmt, test_f, cover_f)

    print("B4 generation complete.")


if __name__ == "__main__":
    main()
