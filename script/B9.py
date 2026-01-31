# Created By: Ryan Wolk (rwolk@hmc.edu) on 1/29/2026

import coverfloat
import random 
import itertools

from typing import Generator

"""
From Ahronhi et al 2008:

B9. Special Significands on Inputs
This model tests special patterns in the significands of the input operands. Each
of the input operands should contain one of the following patterns (each
sequence can be of length 0 up to the number of bits in the significand – the
more interesting cases will be chosen).
i. A sequence of leading zeroes
ii. A sequence of leading ones
iii. A sequence of trailing zeroes
iv. A sequence of trailing ones
v. A small number of 1s as compared to 0s
vi. A small number of 0s as compared to 1s
vii. A "checkerboard" pattern (for example 00110011… or 011011011…)
viii. Long sequences of 1s
ix. Long sequences of 0s
Operation: Divide, Remainder, Square-root, Multiply
Enable Bits: XE
"""

OP_ADD    = "00000010"
OP_SUB    = "00000020"
OP_MUL    = "00000030"
OP_DIV    = "00000040"
OP_FMA    = "00000050"
OP_FMADD  = "00000051"
OP_FMSUB  = "00000052"
OP_FNMADD = "00000053"
OP_FNMSUB = "00000054"
OP_SQRT   = "00000060"
OP_REM    = "00000070"
OP_CFI    = "00000080"
OP_FCVTW  = "00000081"
OP_FCVTWU = "00000082"
OP_FCVTL  = "00000083"
OP_FCVTLU = "00000084"
OP_CFF    = "00000090"
OP_CIF    = "000000A0"
OP_QC     = "000000B0"
OP_FEQ    = "000000B1"
OP_SC     = "000000C0"
OP_FLT    = "000000C1"
OP_FLE    = "000000C2"
OP_CLASS  = "000000D0"
OP_MIN    = "000000E0"
OP_MAX    = "000000F0"
OP_CSN    = "00000100"
OP_FSGNJ  = "00000101"
OP_FSGNJN = "00000102"
OP_FSGNJX = "00000103"

OPS = [
    OP_DIV,
    OP_REM,
    OP_SQRT,
    OP_MUL,
]

FMT_HALF   = "00" # 00000000
FMT_SINGLE = "01" # 00000001
FMT_DOUBLE = "02" # 00000010
FMT_QUAD   = "03" # 00000011
FMT_BF16   = "04" # 00000100

def generate_checkerboards(length: int) -> Generator[str]:
    for zeros_length in range(1, length // 2 + 1):
        for ones_length in range(1, length // 2 + 1):
            pattern = '0' * zeros_length + '1' * ones_length
            pattern *= length // (zeros_length + ones_length) + 1
            yield pattern[:length]
            yield pattern[::-1][:length]

def generate_leading_and_trailing_zeros(length: int) -> Generator[str]:
    for zeros_length in range(1, length):
        pattern = '0' * zeros_length + '1' + bin(random.getrandbits(length))[2:]
        yield pattern[:length]
        yield pattern[:length][::-1]

def generate_leading_and_trailing_ones(length: int) -> Generator[str]:
    for ones_length in range(1, length):
        pattern = '1' * ones_length + '0' + bin(random.getrandbits(length))[2:]
        yield pattern[:length]
        yield pattern[:length][::-1]

def generate_with_k_ones(k: int, length: int) -> Generator[str]:
    combinations = list(itertools.combinations(range(length), k))
    random.shuffle(combinations)

    for combo in combinations:
        pattern = ['0'] * length
        for index in combo:
            pattern[index] = '1'
        yield ''.join(pattern)

def generate_with_long_runs(min_run_length: int, length: int) -> Generator[str]:
    for start in range(length - min_run_length + 1):
        # Generate a pattern with random digits
        pattern = list(bin(random.getrandbits(length))[2:].zfill(length))

        # Fill in a run of ones
        ones_pattern = pattern[:]
        for i in range(start, start + min_run_length):
            ones_pattern[i] = '1'
        yield ''.join(ones_pattern)

        zeros_pattern = pattern[:]
        for i in range(start, start + min_run_length):
            zeros_pattern[i] = '0'
        yield ''.join(zeros_pattern)
    
def main():
    for pattern in generate_checkerboards(16):
        print(pattern)

if __name__ == '__main__':
    main()