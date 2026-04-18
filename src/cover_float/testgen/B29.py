# B29 (rwolk@g.hmc.edu)

import random
from pathlib import Path
from typing import TextIO

import cover_float.common.constants as constants
from cover_float.common.util import generate_float, generate_test_vector, reproducible_hash, unpack_test_vector
from cover_float.reference import run_test_vector, store_cover_vector


def generate_tests(fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    targets = [{"Sign": x & 1, "LSB": (x & 2) >> 1, "Guard": (x & 4) >> 2, "Sticky": (x & 8) >> 3} for x in range(16)]

    nf = constants.MANTISSA_BITS[fmt]
    _exp_min, exp_max = constants.UNBIASED_EXP[fmt]

    for mode in constants.ROUNDING_MODES:
        seed = reproducible_hash(f"B29 {fmt} {mode}")
        random.seed(seed)

        for target in targets:
            exp = random.randint(1, min(nf - 2, exp_max))

            sticky_length = nf - exp - 1
            sticky = random.randint(1, (1 << sticky_length) - 1) if target["Sticky"] else 0
            integer_bits = exp
            integer_part = random.getrandbits(integer_bits - 1) << 1
            if target["LSB"]:
                integer_part |= 1

            sig = ((integer_part << 1) | target["Guard"]) << (sticky_length) | sticky
            f1 = generate_float(target["Sign"], exp, sig, fmt)
            tv = generate_test_vector(constants.OP_RFI, f1, 0, 0, fmt, fmt, mode)

            results = run_test_vector(tv)
            fields = unpack_test_vector(results)

            mantissa = f"{fields.interm_sig:0{constants.INTER_SIGNIFICAND_LENGTH}b}"
            rounding_bits = mantissa[-constants.RFI_DECIMAL_POINT :]

            computed_lsb = int(mantissa[-constants.RFI_DECIMAL_POINT - 1] == "1")
            computed_guard = int(rounding_bits[0] == "1")
            computed_sticky = int(any(x == "1" for x in rounding_bits[1:]))
            computed_sign = fields.interm_sign

            computed_info = {
                "Sign": computed_sign,
                "LSB": computed_lsb,
                "Guard": computed_guard,
                "Sticky": computed_sticky,
            }

            if computed_info == target:
                store_cover_vector(results, test_f, cover_f)
            else:
                print(f"B29 Generation Error: fmt={fmt}, and target={target}")


def main() -> None:
    with (
        Path("tests/testvectors/B29_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B29_cv.txt").open("w") as cover_f,
    ):
        for fmt in constants.FLOAT_FMTS:
            generate_tests(fmt, test_f, cover_f)


if __name__ == "__main__":
    main()
