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


def bezout_inverse(x: int, base: int) -> int:
    # Find the inverse of an element using the Euclidean algorithm and applying Bezout's identity
    # The euclidean algorithm says: gcd(x, y) = gcd(y, x % y) for x > y
    # Bezout's identity says that there exists A, B in Z such that Ax + By = gcd(x, y)
    # With proper book keeping, we can find these X and Y, and noticing that
    # Ax + By = 1 when x, y are relatively prime (as x and base are assumed to be),
    # Ax = 1 - By implies Ax = 1 (mod y) and thus A inverts X in base y

    # Algorithm taken from https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    r = [base, x]
    s = [1, 0]
    t = [0, 1]

    while r[-1] != 0:
        q = r[-2] // r[-1]
        r.append(r[-2] - q * r[-1])
        s.append(s[-2] - q * s[-1])
        t.append(t[-2] - q * t[-1])

    gcd = r[-2]
    _bezout_A = s[-2]
    bezout_B = t[-2]

    if gcd != 1:
        return -1

    # We have bezout_A(base) + bezout_B(x) = gcd
    # So, as shown above, bezout_B inverts x
    return bezout_B % base


def factors_to_bit_width(factors: dict[int, int], target: int, bit_width: int) -> tuple[int, int]:
    usable_factors = [factor for factor, count in factors.items() for _ in range(count)]
    usable_factors.sort(key=lambda x: -x)  # Sort Descending

    def recurse(running_count: int, i: int) -> int:
        last_factor = 0
        for idx, factor in enumerate(usable_factors[i:], i):
            if last_factor == factor:
                continue
            last_factor = factor

            guess = running_count * factor
            if guess.bit_length() == bit_width and (target // guess).bit_length() == bit_width:
                return running_count * factor
            elif guess.bit_length() < bit_width:
                attempt = recurse(guess, idx + 1)
                if attempt != 0:
                    return attempt

        return 0

    res = recurse(1, 0)
    if res == 0:
        return (0, 0)

    return (res, target // res)


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


def extract_rounding_info(cover_vector: str) -> dict[str, int]:
    # fields = cover_vector.split("_")
    fields = unpack_test_vector(cover_vector)
    sgn = fields.interm_sign
    result_fmt = fields.output_format

    # Place in a leading one so that we get all the significant figures possible
    # interm_significand = int("1" + fields[-1], 16)
    # interm_significand = bin(interm_significand)[2:][1:]
    interm_significand = f"{fields.interm_sig:0{constants.INTER_SIGNIFICAND_LENGTH}b}"

    if result_fmt in constants.FLOAT_FMTS:
        mantissa_length = constants.MANTISSA_BITS[result_fmt]
    elif result_fmt in constants.INT_FMTS:
        mantissa_length = constants.INT_MAX_EXPS[result_fmt]
    else:
        raise ValueError(f"Unknown Result Format: {result_fmt}")

    lsb = interm_significand[mantissa_length - 1]
    guard = interm_significand[mantissa_length]
    sticky = interm_significand[mantissa_length + 1 :]

    return {
        "Sign": int(sgn),
        "LSB": int(lsb),
        "Guard": int(guard),
        "Sticky": 1 if any(x == "1" for x in sticky) else 0,
    }
