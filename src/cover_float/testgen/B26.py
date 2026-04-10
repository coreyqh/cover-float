# B26 (rwolk@g.hmc.edu)

import random
from pathlib import Path
from typing import TextIO

import cover_float.common.constants as constants
from cover_float.common.util import generate_test_vector, reproducible_hash
from cover_float.reference import run_and_store_test_vector


def generate_B26(int_fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    bits = constants.INT_MAX_EXPS[int_fmt]

    for float_fmt in constants.FLOAT_FMTS:
        seed = reproducible_hash(f"B26 {int_fmt} {float_fmt}")
        random.seed(seed)
        for msb in range(bits):
            unsigned = 1 << msb | random.getrandbits(msb)
            tv = generate_test_vector(
                constants.OP_CIF, unsigned, 0, 0, int_fmt, float_fmt, random.choice(constants.ROUNDING_MODES)
            )
            run_and_store_test_vector(tv, test_f, cover_f)

            if constants.INT_SIGNED[int_fmt]:
                signed = -(1 << msb | random.getrandbits(msb))
                unsigned = signed & (
                    (1 << constants.INT_SIZES[int_fmt]) - 1
                )  # This gives a twos complement representation
                tv = generate_test_vector(
                    constants.OP_CIF, unsigned, 0, 0, int_fmt, float_fmt, random.choice(constants.ROUNDING_MODES)
                )
                run_and_store_test_vector(tv, test_f, cover_f)


def main() -> None:
    with (
        Path("tests/testvectors/B26_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B26_cv.txt").open("w") as cover_f,
    ):
        for fmt in constants.INT_FMTS:
            generate_B26(fmt, test_f, cover_f)


if __name__ == "__main__":
    main()
