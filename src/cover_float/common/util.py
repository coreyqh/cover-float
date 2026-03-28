from dataclasses import dataclass

import cover_float.common.constants as constants


def generate_test_vector(op: str, in1: int, in2: int, in3: int, fmt1: str, fmt2: str, rnd_mode: str = "00") -> str:
    zero_padding = "0" * 32
    return f"{op}_{rnd_mode}_{in1:032x}_{in2:032x}_{in3:032x}_{fmt1}_{zero_padding}_{fmt2}_00\n"


def generate_float(sign: int, exponent: int, mantissa: int, fmt: str) -> int:
    exponent += constants.BIAS[fmt]
    return (
        (sign << (constants.MANTISSA_BITS[fmt] + constants.EXPONENT_BITS[fmt]))
        | (exponent << constants.MANTISSA_BITS[fmt])
        | mantissa
    )


def reproducible_hash(s: str) -> int:
    """
    Return a simple hash of a string for use as a random seed.

    Python randomizes hashes by default, but we need a repeatable hash for repeatable test cases.
    """
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h


@dataclass
class UnpackedTestVector:
    op: str
    rounding_mode: str
    input1: int
    input2: int
    input3: int
    input_format: str
    result: int
    output_format: str
    flags: int
    interm_sign: int
    interm_exp: int
    interm_sig: int
    fma_pre_addition: int


def unpack_test_vector(tv: str) -> UnpackedTestVector:
    parts = tv.split("_")

    if len(parts) < 8:
        raise ValueError(f"Too Few Parts in Test Vector: {tv}")

    op = parts[0]
    rounding_mode = parts[2]
    input1 = int(parts[2], 16)
    input2 = int(parts[3], 16)
    input3 = int(parts[4], 16)
    input_format = parts[5].upper()
    result = int(parts[6], 16)
    output_format = parts[7].upper()

    flags = int(parts[8], 16) if len(parts) > 8 else 0

    if len(parts) > 9:
        interm_sign = int(parts[9], 16)
        interm_exp = int(parts[10], 16)
        interm_sig = int(parts[11], 16)
        fma_pre_addition = int(parts[12], 16)
    else:
        interm_sign = 0
        interm_exp = 0
        interm_sig = 0
        fma_pre_addition = 0

    return UnpackedTestVector(
        op,
        rounding_mode,
        input1,
        input2,
        input3,
        input_format,
        result,
        output_format,
        flags,
        interm_sign,
        interm_exp,
        interm_sig,
        fma_pre_addition,
    )


def get_rounding_bits(cover_vector: str) -> str:
    return bin(unpack_test_vector(cover_vector).interm_sig)[2:].zfill(constants.INTER_SIGNIFICAND_LENGTH)
