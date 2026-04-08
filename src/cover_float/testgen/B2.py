"""
Angela Zheng (angela20061015@gmail.com)

Created:         April 8, 2026
Last Edited:     April 8, 2026
"""

import random
from pathlib import Path
from random import seed
from typing import Callable, TextIO

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

ZERO = "0" * 32


class FPHelper:
    @staticmethod
    def to_hex(fmt: str, sign: int, exp: int, mant: int) -> str:
        bits = f"{sign:1b}" + f"{exp:0{EXPONENT_BITS[fmt]}b}" + f"{mant:0{MANTISSA_BITS[fmt]}b}"
        return f"{int(bits, 2):032X}"

    @staticmethod
    def ref(op: str, a: str, b: str, c: str, fmt: str) -> str:
        vec = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{ZERO}_{fmt}_00"
        return run_test_vector(vec).split("_")[6]

    @staticmethod
    def store(op: str, a: str, b: str, c: str, fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
        vec = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{ZERO}_{fmt}_00"
        run_and_store_test_vector(vec, test_f, cover_f)


class Generator:
    MAX_TRIES = 10

    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        self.m_bits = MANTISSA_BITS[fmt]
        self.bias = BIAS[fmt]
        self.max_exp = BIASED_EXP[fmt][1]

    def random_fp(self, exp_range: tuple[int, int]) -> str:
        return FPHelper.to_hex(
            self.fmt,
            random.randint(0, 1),
            random.randint(*exp_range),
            random.getrandbits(self.m_bits),
        )

    def random_fp_with_sign(self, exp_range: tuple[int, int], sign: int) -> str:
        return FPHelper.to_hex(
            self.fmt,
            sign,
            random.randint(*exp_range),
            random.getrandbits(self.m_bits),
        )

    def solve_exact(
        self,
        op: str,
        desired: str,
        builder: Callable[[], tuple[str, str, str]],
    ) -> tuple[str, str, str]:

        a, b, c = builder()
        last = (a, b, c)

        for _ in range(self.MAX_TRIES):
            a, b, c = builder()
            res = FPHelper.ref(op, a, b, c, self.fmt)
            if res == desired.lower():
                return a, b, c
            last = (a, b, c)

        a, _, _ = last
        b = FPHelper.ref(OP_SUB, desired, a, ZERO, self.fmt)
        return a, b, ZERO

    def gen_add(self, desired: str, base_e: int, maxnorm: bool, sign: int) -> tuple[str, str, str]:
        neg_zero = FPHelper.to_hex(self.fmt, 1, 0, 0)

        def builder() -> tuple[str, str, str]:
            if desired == neg_zero:
                return neg_zero, neg_zero, ZERO

            a_sign = sign if maxnorm else random.randint(0, 1)

            a = FPHelper.to_hex(
                self.fmt,
                a_sign,
                base_e,
                random.getrandbits(self.m_bits),
            )
            b = FPHelper.ref(OP_SUB, desired, a, ZERO, self.fmt)
            return a, b, ZERO

        return self.solve_exact(OP_ADD, desired, builder)

    def gen_sub(self, desired: str, base_e: int, maxnorm: bool, sign: int) -> tuple[str, str, str]:
        neg_zero = FPHelper.to_hex(self.fmt, 1, 0, 0)
        pos_zero = FPHelper.to_hex(self.fmt, 0, 0, 0)

        def builder() -> tuple[str, str, str]:
            if desired == neg_zero:
                return neg_zero, pos_zero, ZERO

            a_sign = sign if maxnorm else random.randint(0, 1)

            a = FPHelper.to_hex(
                self.fmt,
                a_sign,
                base_e,
                random.getrandbits(self.m_bits),
            )
            b = FPHelper.ref(OP_SUB, a, desired, ZERO, self.fmt)
            return a, b, ZERO

        return self.solve_exact(OP_SUB, desired, builder)

    def gen_mul(self, desired: str, maxnorm: bool, sign: int) -> tuple[str, str, str]:
        exp_range = (self.bias, self.max_exp) if maxnorm else (1, self.bias)

        def builder() -> tuple[str, str, str]:
            a = self.random_fp_with_sign(exp_range, sign) if maxnorm else self.random_fp(exp_range)
            b = FPHelper.ref(OP_DIV, desired, a, ZERO, self.fmt)
            return a, b, ZERO

        return self.solve_exact(OP_MUL, desired, builder)

    def gen_div(self, desired: str, maxnorm: bool, sign: int) -> tuple[str, str, str]:
        exp_range = (self.bias, self.max_exp) if maxnorm else (1, self.bias - self.m_bits)

        def builder() -> tuple[str, str, str]:
            a = self.random_fp_with_sign(exp_range, sign) if maxnorm else self.random_fp(exp_range)
            b = FPHelper.ref(OP_DIV, a, desired, ZERO, self.fmt)
            return a, b, ZERO

        return self.solve_exact(OP_DIV, desired, builder)

    def gen_sqrt(self, desired: str) -> tuple[str, str, str]:
        def builder() -> tuple[str, str, str]:
            a = FPHelper.ref(OP_MUL, desired, desired, ZERO, self.fmt)
            return a, ZERO, ZERO

        return self.solve_exact(OP_SQRT, desired, builder)

    def gen_fma(self, op: str, desired: str, base_e: int, maxnorm: bool) -> tuple[str, str, str]:

        def builder() -> tuple[str, str, str]:
            if maxnorm:
                a_exp = random.randint(self.bias, self.max_exp)
                b_exp = random.randint(
                    max(0, self.max_exp - a_exp - self.m_bits),
                    self.max_exp - a_exp,
                )
            else:
                a_exp = random.randint(1, self.bias)
                b_exp = base_e - a_exp + self.bias

            a = FPHelper.to_hex(self.fmt, random.randint(0, 1), a_exp, random.getrandbits(self.m_bits))
            b = FPHelper.to_hex(self.fmt, random.randint(0, 1), b_exp, random.getrandbits(self.m_bits))

            if op == OP_FMADD:
                c = FPHelper.ref(OP_FNMSUB, a, b, desired, self.fmt)
            elif op == OP_FMSUB:
                c = FPHelper.ref(OP_FMSUB, a, b, desired, self.fmt)
            elif op == OP_FNMADD:
                c = FPHelper.ref(OP_FNMADD, a, b, desired, self.fmt)
            else:
                c = FPHelper.ref(OP_FMADD, a, b, desired, self.fmt)

            return a, b, c

        return self.solve_exact(op, desired, builder)


def main() -> None:
    with (
        Path("./tests/testvectors/B2_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B2_cv.txt").open("w") as cover_f,
    ):
        for fmt in FLOAT_FMTS:
            gen = Generator(fmt)

            bases = {
                "Zero": (0, 0),
                "One": (0, (1 << (EXPONENT_BITS[fmt] - 1)) - 1),
                "MinSub": (1, 0),
                "MaxSub": ((1 << gen.m_bits) - 1, 0),
                "MinNorm": (0, 1),
                "MaxNorm": ((1 << gen.m_bits) - 1, (1 << EXPONENT_BITS[fmt]) - 2),
            }

            for base, (base_m, base_e) in bases.items():
                maxnorm = base == "MaxNorm"

                for i in range(gen.m_bits):
                    for sign in [0, 1]:
                        desired = FPHelper.to_hex(fmt, sign, base_e, base_m ^ (1 << i))
                        seed(reproducible_hash(f"{fmt}_{base}_{i}_{sign}"))

                        a, b, c = gen.gen_add(desired, base_e, maxnorm, sign)
                        FPHelper.store(OP_ADD, a, b, c, fmt, test_f, cover_f)

                        a, b, c = gen.gen_sub(desired, base_e, maxnorm, sign)
                        FPHelper.store(OP_SUB, a, b, c, fmt, test_f, cover_f)

                        a, b, c = gen.gen_mul(desired, maxnorm, sign)
                        FPHelper.store(OP_MUL, a, b, c, fmt, test_f, cover_f)

                        a, b, c = gen.gen_div(desired, maxnorm, sign)
                        FPHelper.store(OP_DIV, a, b, c, fmt, test_f, cover_f)

                        if sign == 0 and base == "One":
                            a, b, c = gen.gen_sqrt(desired)
                            FPHelper.store(OP_SQRT, a, b, c, fmt, test_f, cover_f)

                        for op in [OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]:
                            a, b, c = gen.gen_fma(op, desired, base_e, maxnorm)
                            FPHelper.store(op, a, b, c, fmt, test_f, cover_f)

                # for -0 cases
                if base == "Zero":
                    desired = FPHelper.to_hex(fmt, 1, 0, 0)

                    seed(reproducible_hash(f"{fmt}_zero_add_1"))
                    a, b, c = gen.gen_add(desired, 0, False, 1)
                    FPHelper.store(OP_ADD, a, b, c, fmt, test_f, cover_f)

                    seed(reproducible_hash(f"{fmt}_zero_sub_1"))
                    a, b, c = gen.gen_sub(desired, 0, False, 1)
                    FPHelper.store(OP_SUB, a, b, c, fmt, test_f, cover_f)


if __name__ == "__main__":
    main()
