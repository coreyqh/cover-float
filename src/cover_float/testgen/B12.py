"""
Angela Zheng

February 10, 2026

SUMMARY
This script generates cancellation test vectors for the B12 model:

    For the difference d between the exponent of the intermediate result
    and the maximum exponent of the inputs:
        d ∈ [-p, +1]
    Enable Bits: XE

DEFINITION
p: precision of the format, including the hidden 1. (# mantissa bits + 1)
a: first operand
b: second operand
d: the difference between max(a_exp, b_exp) and exponent of the intermediate result

Total test vectors generated: 438
"""
# TODO: For future: implement logic to get different a and b exponents in regular cases

import random

from cover_float.reference import run_and_store_test_vector
from cover_float.common.constants import *

vector_count = 0

TEST_VECTOR_WIDTH_HEX = 144

OP_ADD = "00000010"
OP_SUB = "00000020"

ROUND_NEAR_EVEN = "00"

FMT_HALF   = "00"
FMT_SINGLE = "01"
FMT_DOUBLE = "02"
FMT_QUAD   = "03"
FMT_BF16   = "04"

FMTS = [FMT_SINGLE, FMT_DOUBLE, FMT_QUAD, FMT_HALF, FMT_BF16]

MANTISSA_BITS = {
    FMT_HALF: 10,
    FMT_SINGLE: 23,
    FMT_DOUBLE: 52,
    FMT_QUAD: 112,
    FMT_BF16: 7
}

EXPONENT_BITS = {
    FMT_HALF: 5,
    FMT_SINGLE: 8,
    FMT_DOUBLE: 11,
    FMT_QUAD: 15,
    FMT_BF16: 8
}

BIASED_EXP = {
    FMT_HALF:   [1, 30],
    FMT_SINGLE: [1, 254],
    FMT_DOUBLE: [1, 2046],
    FMT_QUAD:   [1, 32766],
    FMT_BF16:   [1, 254]
}

def decimalComponentsToHex(fmt, sign, biased_exp, mantissa):
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"

    
def write_add(fmt, a_hex, b_hex, test_f, cover_f):
    run_and_store_test_vector(f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n", test_f, cover_f)


def write_sub(fmt, a_hex, b_hex, test_f, cover_f):
    run_and_store_test_vector(f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n", test_f, cover_f)


def makeCancellationMantissas(fmt, d):
    """
    Generate identical -d bits for both operands such that exactly -d bits cancel.
    """

    # d = -8, k = 8
    # a = 01011011 11 1110111111001    8 random bits + 11 + m-k-2 random bits
    # b = 01011011 00 1110111111001    same 8 random bits + 00 + same m-k-2 random bits
    # The 11 and 00 are to prevent borrowing from the previous bit canceling the differing bit

    m = MANTISSA_BITS[fmt]
    k = -d

    # prefix for a and b
    if d == 0:
        tail = random.getrandbits(m-2)
        a_m = 1 << (m-1) | 1 << (m-2) | tail
        b_m = 0 << (m-1) | 0 << (m-2) | tail
        return a_m, b_m
    else:
        a_prefix = 1 << (m-1) | random.getrandbits(k-1) << (m - k)
        b_prefix = a_prefix

    # differing bit
    diff_bit = 1 << (m - k - 1)

    # tails
    if k < (m-1):
        a_tail = 1 << (m - k - 2) | random.getrandbits(m - k - 2)
        b_tail = random.getrandbits(m - k - 2)
    else:
        a_tail = 0
        b_tail = 0

    a_m = a_prefix | diff_bit | a_tail
    b_m = b_prefix | b_tail

    return a_m, b_m


def makeExactCancelMantissas(fmt):
    '''
    Generate mantissas so that exactly m bits cancel.
    '''

    # a = 11011011011110111111001 0     0 + 21 random bits (identical) + ends in 1
    # b = 101101101111011111100 01      same 21 bits (identical) + ends in 01
    # b = 11011011011110111111000 1     after b shifts right, it is in alignment to cancel m bits

    m = MANTISSA_BITS[fmt]

    identical = random.getrandbits(m-2)
    a_m = 1 << (m-1) | identical << 1 | 1
    b_m = identical | 0 << 1 | 1

    return a_m, b_m


def makeCarryMantissas(fmt):
    """
    Force carry for d = +1
    """

    m = MANTISSA_BITS[fmt]

    a_m = (1 << m) - 1          # 1.111...111
    b_m = 1 << (m - 1) | 1      # 1.000...001 (LSB set)

    return a_m, b_m


def makeNegPMantissas(fmt):
    """
    Shifts b exp down 1 to create cancellation of p bits.
    """
    m = MANTISSA_BITS[fmt]
    
    a_m = 0                 # A = 1.00...0 (Mantissa 0)
    b_m = (1 << m) - 1      # B = 1.11...1 (Mantissa all 1s), b_exp = a_exp - 1
    
    return a_m, b_m


def makeTestVectors(fmt, d, operation, test_f, cover_f):
    m = MANTISSA_BITS[fmt]
    p = m + 1
    min_exp, max_exp = BIASED_EXP[fmt]

    is_carry = False
    is_add = (operation == "add")
    write_fn = write_add if is_add else write_sub

    # Exponents
    a_exp = random.randint(min_exp-d, max_exp)
    b_exp = a_exp

    # Mantissas
    if d == 1:
        is_carry = True
        a_m, b_m = makeCarryMantissas(fmt)
    elif d == -p:
        a_m, b_m = makeNegPMantissas(fmt)
        b_exp -= 1
    elif d == -m:
        a_m, b_m = makeExactCancelMantissas(fmt)
        b_exp -= 1
    else:
        a_m, b_m = makeCancellationMantissas(fmt, d)

    # Signs
    if is_add:
        if is_carry:
            a_sign = 0
            b_sign = 0
        else:
            a_sign = 0
            b_sign = 1
    else:
        if is_carry:
            a_sign = 0
            b_sign = 1
        else:
            a_sign = 0
            b_sign = 0

    a_hex = decimalComponentsToHex(fmt, a_sign, a_exp, a_m)
    b_hex = decimalComponentsToHex(fmt, b_sign, b_exp, b_m)

    write_fn(fmt, a_hex, b_hex, test_f, cover_f)
    

def CancellationTests(test_f, cover_f, fmt):
    p = MANTISSA_BITS[fmt] + 1

    for d in range(-p, 2):  # [-p, +1]
        makeTestVectors(fmt, d, "add", test_f, cover_f)
        makeTestVectors(fmt, d, "sub", test_f, cover_f)


def main():
    with open("./tests/testvectors/B12_tv.txt", "w") as test_f, open("./tests/covervectors/B12_cv.txt", "w") as cover_f:
        test_f.write("// Cancellation tests\n")
        test_f.write("// Operations: ADD, SUB\n")

        for fmt in FMTS:
            CancellationTests(test_f, cover_f, fmt)


if __name__ == "__main__":
    main()
