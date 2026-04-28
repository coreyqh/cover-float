"""
Angela Zheng (angela20061015@gmail.com)

B16. Multiply-Add: Cancellation
This model tests every possible value for cancellation.
For the difference between the exponent of the intermediate result and the
maximum between the exponents of the addend and the multiplication result,
test all values in the range:
 [-(2 * p + 1), 1].

My plan:
For each of the fmadd, fmsub, fnmadd, fnmsub operations:

We must ensure that a_exp is the largest exp out of the
three operands because with +c alone would only be able to cancel -p. So,
randomly generate a_exp, and generate b_exp (would probably be negative) so that b_exp = d
and make c_exp = a_exp + b_exp and generate a_m, b_m, and c_m so that they don't result in carry or
more cancellation.
"""

import random
from pathlib import Path
from random import seed
from typing import TextIO

from cover_float.common.constants import (
    BIAS,
    EXPONENT_BITS,
    FLOAT_FMTS,
    MANTISSA_BITS,
    OP_FMADD,
    OP_FMSUB,
    OP_FNMADD,
    OP_FNMSUB,
    OP_MUL,
    UNBIASED_EXP,
)
from cover_float.common.util import (
    decimal_components_to_hex,
    generate_test_vector,
    get_result_from_ref,
    reproducible_hash,
)
from cover_float.reference import run_and_store_test_vector

OPS = [OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]
SOLVER_OPS = {
    OP_FMADD: OP_FNMSUB,
    OP_FMSUB: OP_FMSUB,
    OP_FNMADD: OP_FNMADD,
    OP_FNMSUB: OP_FMADD,
}


def extract_unbiased_exp(fp_hex: str, fmt: str) -> int:
    bits = int(fp_hex, 16)
    exp_bits = EXPONENT_BITS[fmt]
    mant_bits = MANTISSA_BITS[fmt]
    bias = BIAS[fmt]
    exp_mask = (1 << exp_bits) - 1
    exp = (bits >> mant_bits) & exp_mask
    return exp - bias


def generate_deep_cancel(fmt: str, d: int, op: str, test_f: TextIO, cover_f: TextIO) -> bool:
    bias = BIAS[fmt]
    min_raw, max_raw = UNBIASED_EXP[fmt]

    a_sign, b_sign = random.randint(0, 1), random.randint(0, 1)

    # Force Result to EXACTLY 0. Zero has an exponent of (min_raw - 1)
    res_raw = min_raw - 1
    res_m = 0
    res_sign = 0
    res_hex = decimal_components_to_hex(fmt, res_sign, res_raw + bias, res_m)

    # Calculate required product exponent to achieve depth d relative to 0
    target_prod_exp = res_raw - d

    a_raw_min = max(min_raw, target_prod_exp - max_raw)
    a_raw_max = min(max_raw, target_prod_exp - min_raw)

    if a_raw_min > a_raw_max:
        return False

    a_raw = random.randint(a_raw_min, a_raw_max)
    b_raw = target_prod_exp - a_raw

    # Keep mantissas 0 so A*B is exactly representable, guaranteeing the solver succeeds
    a_m, b_m = 0, 0

    a_hex = decimal_components_to_hex(fmt, a_sign, a_raw + bias, a_m)
    b_hex = decimal_components_to_hex(fmt, b_sign, b_raw + bias, b_m)

    try:
        c_hex = get_result_from_ref(SOLVER_OPS[op], a_hex, b_hex, res_hex, fmt)
        vector = generate_test_vector(op, int(a_hex, 16), int(b_hex, 16), int(c_hex, 16), fmt, fmt)
        run_and_store_test_vector(vector, test_f, cover_f)
        return True
    except Exception:
        return False


# Maybe see whether ab_exp is greater than c_exp
# force c_exp to be greater than ab_exp
def generate_same_exp(fmt: str, d: int, op: str, test_f: TextIO, cover_f: TextIO) -> bool:
    m = MANTISSA_BITS[fmt]
    bias = BIAS[fmt]
    max_raw = UNBIASED_EXP[fmt][1]

    a_sign, b_sign = random.randint(0, 1), random.randint(0, 1)
    # res_sign = a_sign ^ b_sign
    c_sign = (a_sign ^ b_sign) if op in [OP_FMADD, OP_FNMADD] else not (a_sign ^ b_sign)

    # We have to make sure c_exp is the greatest, so a_exp and b_exp must both be positive
    target_prod_exp = random.randint(0, max_raw)
    a_raw = random.randint(0, target_prod_exp)
    b_raw = target_prod_exp - a_raw
    c_raw = a_raw + b_raw + 3

    a_m, b_m, c_m = random.getrandbits(m), random.getrandbits(m), random.getrandbits(m)

    a_hex = decimal_components_to_hex(fmt, a_sign, a_raw + bias, a_m)
    b_hex = decimal_components_to_hex(fmt, b_sign, b_raw + bias, b_m)

    # prod_hex = get_result_from_ref(OP_MUL, a_hex, b_hex, "0", fmt)
    # ab_exp = extract_unbiased_exp(prod_hex, fmt)
    # c_raw = ab_exp + 2

    # if not (min_raw <= res_raw <= max_raw):
    #     return False

    # res_hex = decimal_components_to_hex(fmt, res_sign, res_raw + bias, res_m)

    # c_hex = get_result_from_ref(SOLVER_OPS[op], a_hex, b_hex, res_hex, fmt)
    c_hex = decimal_components_to_hex(fmt, c_sign, c_raw + bias, c_m)

    prod_hex = get_result_from_ref(op, a_hex, b_hex, c_hex, fmt)
    prod_exp = extract_unbiased_exp(prod_hex, fmt)
    if prod_exp != c_raw:
        return False
    vector = generate_test_vector(op, int(a_hex, 16), int(b_hex, 16), int(c_hex, 16), fmt, fmt)
    run_and_store_test_vector(vector, test_f, cover_f)
    return True


def generate_carry(fmt: str, d: int, op: str, test_f: TextIO, cover_f: TextIO) -> bool:
    m = MANTISSA_BITS[fmt]
    bias = BIAS[fmt]
    max_raw = UNBIASED_EXP[fmt][1]

    a_m, b_m = random.getrandbits(m), random.getrandbits(m)
    c_m = (1 << m) - 1

    # Exponents are guarded against overflow by dividing max exponent by two to
    # account for that the intermediate product exponent is a_raw + b_raw. But as d = 0, both
    # a_exp and b_exp need to be positive to make C greatest
    a_raw = random.randint(0, (max_raw - 1) // 2)
    b_raw = random.randint(0, (max_raw - 1) // 2)
    c_raw = a_raw + b_raw + 1

    a_sign = random.randint(0, 1)
    b_sign = a_sign
    c_sign = 0 if op in [OP_FMADD, OP_FNMADD] else 1

    a_hex = decimal_components_to_hex(fmt, a_sign, a_raw + bias, a_m)
    b_hex = decimal_components_to_hex(fmt, b_sign, b_raw + bias, b_m)
    c_hex = decimal_components_to_hex(fmt, c_sign, c_raw + bias, c_m)

    vector = generate_test_vector(op, int(a_hex, 16), int(b_hex, 16), int(c_hex, 16), fmt, fmt)
    run_and_store_test_vector(vector, test_f, cover_f)
    return True


def generate_standard(fmt: str, d: int, op: str, test_f: TextIO, cover_f: TextIO) -> bool:
    m = MANTISSA_BITS[fmt]
    bias = BIAS[fmt]
    min_raw, max_raw = UNBIASED_EXP[fmt]

    a_sign, b_sign = random.randint(0, 1), random.randint(0, 1)
    res_sign = random.randint(0, 1)

    valid_min_prod = max(min_raw, (min_raw - 1) - d)
    valid_max_prod = min(max_raw, max_raw - d)

    if valid_min_prod > valid_max_prod:
        return False

    target_prod_exp = random.randint(valid_min_prod, valid_max_prod)

    a_raw_min = max(min_raw, target_prod_exp - max_raw)
    a_raw_max = min(max_raw, target_prod_exp - min_raw)

    if a_raw_min > a_raw_max:
        return False

    a_raw = random.randint(a_raw_min, a_raw_max)
    b_raw = target_prod_exp - a_raw

    if d < -m:
        target_depth = abs(d)
        sum_kj = max(0, 2 * m - target_depth)
        k = sum_kj // 2
        j = sum_kj - k
        a_m = 1 << k
        b_m = 1 << j
        res_m = 0
    else:
        a_m, b_m, res_m = random.getrandbits(m), random.getrandbits(m), random.getrandbits(m)

    a_hex = decimal_components_to_hex(fmt, a_sign, a_raw + bias, a_m)
    b_hex = decimal_components_to_hex(fmt, b_sign, b_raw + bias, b_m)

    prod_hex = get_result_from_ref(OP_MUL, a_hex, b_hex, "0", fmt)
    ab_exp = extract_unbiased_exp(prod_hex, fmt)

    res_raw = ab_exp + d

    if not (min_raw - 1 <= res_raw <= max_raw):
        return False

    res_hex = decimal_components_to_hex(fmt, res_sign, res_raw + bias, res_m)

    try:
        c_hex = get_result_from_ref(SOLVER_OPS[op], a_hex, b_hex, res_hex, fmt)
        vector = generate_test_vector(op, int(a_hex, 16), int(b_hex, 16), int(c_hex, 16), fmt, fmt)
        run_and_store_test_vector(vector, test_f, cover_f)
        return True
    except Exception:
        return False


def main() -> None:
    with (
        Path("./tests/testvectors/B16_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B16_cv.txt").open("w") as cover_f,
    ):
        for fmt in FLOAT_FMTS:
            p = MANTISSA_BITS[fmt] + 1
            for d in range(-(2 * p + 1), 2):
                for op in OPS:
                    seed(reproducible_hash(f"{fmt}_b16_{d}_{op}"))

                    max_retries = 5
                    for _ in range(max_retries):
                        success = False

                        if d <= -(2 * p - 1):
                            success = generate_deep_cancel(fmt, d, op, test_f, cover_f)
                        elif d == 0:
                            success = generate_same_exp(fmt, d, op, test_f, cover_f)
                        elif d == 1:
                            success = generate_carry(fmt, d, op, test_f, cover_f)
                        else:
                            success = generate_standard(fmt, d, op, test_f, cover_f)
                        if success:
                            break


if __name__ == "__main__":
    main()
