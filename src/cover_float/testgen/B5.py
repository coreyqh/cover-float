# Lamarr
# B5 Model


import random
from typing import TextIO

from cover_float.common.constants import (
    EXPONENT_BIAS,
    EXPONENT_BITS,
    FMT_BF16,
    FMT_DOUBLE,
    FMT_HALF,
    FMT_QUAD,
    FMT_SINGLE,
    MANTISSA_BITS,
    OP_CFF,
    ROUNDING_MODES,
    UNBIASED_EXP,
)
from cover_float.reference import run_and_store_test_vector

B5_FMTS = [FMT_QUAD, FMT_DOUBLE, FMT_SINGLE, FMT_BF16, FMT_HALF]


def generate_FP(
    input_e_bitwidth: int,
    input_sign: str,
    input_exponent: int,
    input_mantissa: str,
    input_bias: int
    ) -> str:
    exponent = f"{input_exponent + input_bias:0{input_e_bitwidth}b}"
    complete = input_sign + exponent + input_mantissa
    fp_complete = format(int(complete, 2), "X")

    return fp_complete


def tests_conversion_1_2(
    lp: str,
    hp: str,
    rounding_mode: str,
    test_f: TextIO,
    cover_f: TextIO
    ) -> None:
    hp_m_bits = MANTISSA_BITS[hp]
    hp_e_bits = EXPONENT_BITS[hp]
    hp_e_bias = EXPONENT_BIAS[hp]
    lp_sn_exp = UNBIASED_EXP[lp][0] - 1  # Make the exponent for subnorm values
    lp_m_bits = MANTISSA_BITS[lp]

    hp_max_exp = lp_sn_exp - 1
    hp_min_exp = lp_sn_exp - lp_m_bits

    if hp == FMT_SINGLE and lp == FMT_BF16:  # Different case for FP_32 and BF_16
        hp_min_exp = lp_sn_exp
        hp_max_exp = lp_sn_exp

    input_1_exponent = random.randint(hp_min_exp, hp_max_exp)
    input_2_exponent = random.randint(hp_min_exp, hp_max_exp)

    max_mantissa = int("1" * hp_m_bits, 2)

    input_1_mantissa = f"{random.randint(0, max_mantissa):0{hp_m_bits}b}"
    input_2_mantissa = f"{random.randint(0, max_mantissa):0{hp_m_bits}b}"

    input_value_1 = generate_FP(hp_e_bits, "0", input_1_exponent, input_1_mantissa, hp_e_bias)
    input_value_2 = generate_FP(hp_e_bits, "1", input_2_exponent, input_2_mantissa, hp_e_bias)

    run_and_store_test_vector(
        f"{OP_CFF}_{rounding_mode}_{input_value_1}_{32 * '0'}_{32 * '0'}_{hp}_{32 * '0'}_{lp}_00", test_f, cover_f
    )  # Test 1
    run_and_store_test_vector(
        f"{OP_CFF}_{rounding_mode}_{input_value_2}_{32 * '0'}_{32 * '0'}_{hp}_{32 * '0'}_{lp}_00", test_f, cover_f
    )  # Test 2


def genPNTestVectors(
    lp: str,
    hp: str,
    rounding_mode: str,
    hp_e_bits: int,
    hp_exp: int,
    complete_binary_1: str,
    complete_binary_2: str,
    hp_e_bias: int,
    test_f: str,
    cover_f: str
) -> None:
    input_value_1 = generate_FP(hp_e_bits, "0", hp_exp, complete_binary_1, hp_e_bias)
    input_value_2 = generate_FP(hp_e_bits, "1", hp_exp, complete_binary_2, hp_e_bias)

    run_and_store_test_vector(
        f"{OP_CFF}_{rounding_mode}_{input_value_1}_{32 * '0'}_{32 * '0'}_{hp}_{32 * '0'}_{lp}_00", test_f, cover_f
    )  # Test 1
    run_and_store_test_vector(
        f"{OP_CFF}_{rounding_mode}_{input_value_2}_{32 * '0'}_{32 * '0'}_{hp}_{32 * '0'}_{lp}_00", test_f, cover_f
    )  # Test 2


def tests_conversion_3_4(
    lp: str,
    hp: str,
    rounding_mode: str,
    test_f: str,
    cover_f: str
    ) -> None:
    hp_m_bits = MANTISSA_BITS[hp]
    hp_e_bits = EXPONENT_BITS[hp]
    hp_e_bias = EXPONENT_BIAS[hp]
    lp_sn_exp = UNBIASED_EXP[lp][0] - 1  # Account for subnorms
    lp_m_bits = MANTISSA_BITS[lp]

    hp_sn_lp_exp = lp_sn_exp - lp_m_bits

    if hp == FMT_SINGLE and lp == FMT_BF16:
        # We can't perform the same exponent operations when hp = lp in single -> bf 16
        # the desired values of the gaurd, round, and sticky bits:
        # minSN -3 i_ulp, minSN - 2 i_ulp, minSN - 1 i_ulp, minSN, minSN + 1 i_ulp, minSN + 2 i_ulp, minSN + 3 i_ulp
        grs = ["001", "010", "011", "100", "101", "110", "111"]

        leading_zeros_len = lp_m_bits - 1
        determined_len = leading_zeros_len + 3
        remaining_rand_len = hp_m_bits - determined_len
        max_remaining_rand = int("1" * remaining_rand_len, 2)

        for bits in grs:
            remaining_rand_bits_1 = f"{random.randint(0, max_remaining_rand):0{remaining_rand_len}b}"
            full_mantissa_1 = bits + remaining_rand_bits_1

            remaining_rand_bits_2 = f"{random.randint(0, max_remaining_rand):0{remaining_rand_len}b}"
            full_mantissa_2 = bits + remaining_rand_bits_2

            genPNTestVectors(
                lp,
                hp,
                rounding_mode,
                hp_e_bits,
                lp_sn_exp,
                full_mantissa_1,
                full_mantissa_2,
                hp_e_bias,
                test_f,
                cover_f,
            )
    else:
        # MinSN, MinSN + 1 ulp, MinSN + 2 ulp, MinSN + 3 ulp
        for i in range(0, int("11", 2) + 1):  # Iterate over differt round and sticky bits
            hp_exp = hp_sn_lp_exp
            rs = f"{i:02b}"  # round and sticky bit

            remaining_mantissa_bits = (
                hp_m_bits - 2
            )  # find out what part of the mantissa bits left need to be randomized
            max_mantissa = int("1" * remaining_mantissa_bits, 2)

            complete_binary_1 = rs + f"{random.randint(0, max_mantissa):0{remaining_mantissa_bits}b}"
            complete_binary_2 = rs + f"{random.randint(0, max_mantissa):0{remaining_mantissa_bits}b}"

            genPNTestVectors(
                lp,
                hp,
                rounding_mode,
                hp_e_bits,
                hp_exp,
                complete_binary_1,
                complete_binary_2,
                hp_e_bias,
                test_f,
                cover_f,
            )

        # MinSN - 1 ulp, MinSN - 2 ulp
        for i in range(0, int("1", 2) + 1):
            hp_exp = hp_sn_lp_exp - 1
            s = f"{i:01b}"  # sticky bit
            remaining_mantissa_bits = hp_m_bits - 1

            max_mantissa = int("1" * remaining_mantissa_bits, 2)

            complete_binary_1 = (s + f"{random.randint(0, max_mantissa):0{remaining_mantissa_bits}b}")
            complete_binary_2 = s + f"{random.randint(0, max_mantissa):0{remaining_mantissa_bits}b}"

            genPNTestVectors(
                lp,
                hp,
                rounding_mode,
                hp_e_bits,
                hp_exp,
                complete_binary_1,
                complete_binary_2,
                hp_e_bias,
                test_f,
                cover_f,
            )

        # #MinSN - 3 ulp
        hp_exp = hp_sn_lp_exp - 2
        remaining_mantissa_bits = hp_m_bits
        max_mantissa = int("1" * hp_m_bits, 2)

        complete_binary_1 = f"{random.randint(0, max_mantissa):0{remaining_mantissa_bits}b}"
        complete_binary_2 = f"{random.randint(0, max_mantissa):0{remaining_mantissa_bits}b}"

        genPNTestVectors(
            lp, hp, rounding_mode, hp_e_bits, hp_exp, complete_binary_1, complete_binary_2, hp_e_bias, test_f, cover_f
        )


def tests_conversion_5_6(
    lp: str,
    hp: str,
    rounding_mode: str,
    test_f: str,
    cover_f: str
    ) -> None:
    hp_m_bits = MANTISSA_BITS[hp]
    hp_e_bits = EXPONENT_BITS[hp]
    hp_e_bias = EXPONENT_BIAS[hp]
    lp_n_exp = UNBIASED_EXP[lp][0]
    lp_sn_exp = UNBIASED_EXP[lp][0] - 1  # Account for subnorms
    lp_m_bits = MANTISSA_BITS[lp]

    if (
        hp != FMT_BF16 and lp != FMT_SINGLE
    ):  # The mantissa bits for bf_16 are smaller than that for single, so you can't do these operations
        rem_bits = hp_m_bits + 1 - 2 - lp_m_bits

        max_rem = int("1" * rem_bits, 2)

        # MinNorm - 1 i_ulp:
        hp_m_1 = "1" * (lp_m_bits - 1) + "1" + "1" + f"{random.randint(1, max_rem):0{rem_bits}b}"
        hp_m_2 = "1" * (lp_m_bits - 1) + "1" + "1" + f"{random.randint(1, max_rem):0{rem_bits}b}"
        hp_exp = lp_sn_exp

        genPNTestVectors(lp, hp, rounding_mode, hp_e_bits, hp_exp, hp_m_1, hp_m_2, hp_e_bias, test_f, cover_f)

        # MinNorm - 2 i_ulp:
        hp_m = "1" * (lp_m_bits - 1) + "1" + "1" + "0" * rem_bits
        hp_exp = lp_sn_exp

        genPNTestVectors(lp, hp, rounding_mode, hp_e_bits, hp_exp, hp_m, hp_m, hp_e_bias, test_f, cover_f)

        # MinNorm - 3 i_ulp:
        hp_m = "1" * (lp_m_bits - 1) + "1" + "0" + f"{random.randint(1, max_rem):0{rem_bits}b}"
        hp_exp = lp_sn_exp

        genPNTestVectors(lp, hp, rounding_mode, hp_e_bits, hp_exp, hp_m, hp_m, hp_e_bias, test_f, cover_f)

        # MinNorm + 1 i_ulp:
        hp_m = "0" * (lp_m_bits - 1) + "0" + "0" + f"{random.randint(1, max_rem):0{rem_bits}b}"
        hp_exp = lp_n_exp

        genPNTestVectors(lp, hp, rounding_mode, hp_e_bits, hp_exp, hp_m, hp_m, hp_e_bias, test_f, cover_f)

        # MinNorm + 2 i_ulp:
        hp_m = "0" * (lp_m_bits - 1) + "0" + "1" + "0" * rem_bits
        hp_exp = lp_n_exp

        genPNTestVectors(lp, hp, rounding_mode, hp_e_bits, hp_exp, hp_m, hp_m, hp_e_bias, test_f, cover_f)

        # MinNorm + 3 i_ulp:
        hp_m = "0" * (lp_m_bits - 1) + "0" + "1" + f"{random.randint(1, max_rem):0{rem_bits}b}"
        hp_exp = lp_n_exp

        genPNTestVectors(lp, hp, rounding_mode, hp_e_bits, hp_exp, hp_m, hp_m, hp_e_bias, test_f, cover_f)


def tests_conversion_7_8(
    lp: str,
    hp: str,
    rounding_mode: str,
    test_f: str,
    cover_f: TextIO
    ) -> None:
    hp_m_bits = MANTISSA_BITS[hp]
    hp_e_bits = EXPONENT_BITS[hp]
    hp_e_bias = EXPONENT_BIAS[hp]
    lp_sn_exp = UNBIASED_EXP[lp][0] - 1
    hp_sn_exp = UNBIASED_EXP[hp][0] - 1
    lp_m_bits = MANTISSA_BITS[lp]

    # Different for fp_16 and bf_16 because they have the same exponent
    if hp == FMT_SINGLE and lp == FMT_BF16:
        hp_exp = hp_sn_exp
        leading_zero_bits = lp_m_bits
        leading_zeros = "0" * lp_m_bits
        remaining_bits = hp_m_bits - lp_m_bits
        max_remaining_mantissa = int("1" * remaining_bits, 2)

        remaining_binary_1 = f"{random.randint(max_remaining_mantissa, max_remaining_mantissa):0{remaining_bits}b}"
        remaining_binary_2 = f"{random.randint(max_remaining_mantissa, max_remaining_mantissa):0{remaining_bits}b}"

        complete_binary_1 = leading_zeros + remaining_binary_1
        complete_binary_2 = leading_zeros + remaining_binary_2

        genPNTestVectors(
            lp, hp, rounding_mode, hp_e_bits, hp_exp, complete_binary_1, complete_binary_2, hp_e_bias, test_f, cover_f
        )
    else:
        hp_max_exp = lp_sn_exp - lp_m_bits - 1  # put the hidden 1 of the hp in the rounding bit of the lp
        hp_min_exp = hp_sn_exp
        max_remaining_mantissa = int("1" * hp_m_bits, 2)

        hp_exp = random.randint(hp_min_exp, hp_max_exp)

        complete_binary = f"{random.randint(0, max_remaining_mantissa):0{hp_m_bits}b}"

        genPNTestVectors(
            lp, hp, rounding_mode, hp_e_bits, hp_exp, complete_binary, complete_binary, hp_e_bias, test_f, cover_f
        )


def tests_conversion_9(lp, hp, rounding_mode, test_f, cover_f):
    hp_m_bits = MANTISSA_BITS[hp]
    hp_e_bits = EXPONENT_BITS[hp]
    hp_e_bias = EXPONENT_BIAS[hp]
    lp_sn_exp = UNBIASED_EXP[lp][0] - 1  # Account for subnorms
    lp_m_bits = MANTISSA_BITS[lp]
    max_m_value = int("1" * hp_m_bits, 2)

    hp_exp = lp_sn_exp

    for i in range(0, 6):
        complete_binary = f"{random.randint(0, max_m_value):0{hp_m_bits}b}"

        input_value_1 = generate_FP(hp_e_bits, f"{random.randint(0, 1)}", hp_exp, complete_binary, hp_e_bias)
        run_and_store_test_vector(
            f"{OP_CFF}_{rounding_mode}_{input_value_1}_{32 * '0'}_{32 * '0'}_{hp}_{32 * '0'}_{lp}_00", test_f, cover_f
        )  # Test 1
        hp_exp += 1


def convertTests(test_f, cover_f):
    # All conversion tests:
    for i_hp in range(len(B5_FMTS)):
        hp = B5_FMTS[i_hp]
        for i_lp in range(i_hp + 1, len(B5_FMTS)):
            lp = B5_FMTS[i_lp]
            for rounding_mode in ROUNDING_MODES:
                tests_conversion_1_2(lp, hp, rounding_mode, test_f, cover_f)
                tests_conversion_3_4(lp, hp, rounding_mode, test_f, cover_f)
                tests_conversion_5_6(lp, hp, rounding_mode, test_f, cover_f)
                tests_conversion_7_8(lp, hp, rounding_mode, test_f, cover_f)
                tests_conversion_9(lp, hp, rounding_mode, test_f, cover_f)


def main():
    with open("./tests/testvectors/B5_tv.txt", "w") as test_f, open("./tests/covervectors/B5_cv.txt", "w") as cover_f:
        convertTests(test_f, cover_f)


if __name__ == "__main__":
    main()
