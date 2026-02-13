# Lamarr Olive - exponent generation
# Ryan Wolk - Significand generation
# B11 Model
# Shift Combined with Special Significands
# In this model we test the combination of different shift values 
# between the inputs, with special patterns in the significands of
# the inputs. Special Significands:
# i. A sequence of leading zeroes
# ii. A sequence of leading ones
# iii. A sequence of trailing zeroes
# iv. A sequence of trailing ones
# v. A small number of 1s as compared to 0s
# vi. A small number of 0s as compared to 1s
# vii. A "checkerboard" pattern (for example 00110011… or 011011011…)
# viii. Long sequences of 1s
# ix. Long sequences of 0s

import random 
import itertools
import subprocess

from cover_float.reference import run_and_store_test_vector
from cover_float.common.util import generate_test_vector

from typing import Generator, List, TextIO

from cover_float.reference import run_and_store_test_vector
from cover_float.common.constants import *

FMT_INVAL  = "FF" # 11111111
FMT_HALF   = "00" # 00000000
FMT_SINGLE = "01" # 00000001
FMT_DOUBLE = "02" # 00000010
FMT_QUAD   = "03" # 00000011
FMT_BF16   = "04" # 00000100
FMT_INT    = "81" # 10000001
FMT_UINT   = "C1" # 11000001
FMT_LONG   = "82" # 10000010
FMT_ULONG  = "C2" # 11000010

FLOAT_FMTS = [FMT_SINGLE, FMT_DOUBLE, FMT_QUAD, FMT_HALF, FMT_BF16]


def decimalComponentsToHex(fmt, sign, b_mantissa, biased_exp):
    b_sign = f"{sign:01b}"
    b_exponent = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_complete = b_sign + b_exponent + b_mantissa
    h_complete = f"{int(b_complete, 2):032X}"
    return h_complete

def innerTest(fmt):
    p = MANTISSA_BITS[fmt] + 1
    min_exp = BIASED_EXP[fmt][0]
    max_exp = BIASED_EXP[fmt][1]
        
    exponents_list = []
       
        #Incrementing b_exp
       
    a_exp = random.randint(min_exp , max_exp - (p+4))
    b_exp = a_exp
       
    for i in range(0, p+5):
        exponents_list.append([a_exp, b_exp])

        b_exp +=1
             
        #Decrementing b_exp
       
    a_exp = random.randint(min_exp + (p + 4), max_exp)
    b_exp = a_exp-1
       
    for i in range(0, p+4):
        exponents_list.append([a_exp, b_exp])
            
        b_exp -=1 #Final statement, decrements 1 under
    return exponents_list
    
   
def outerTest(fmt):
    p = MANTISSA_BITS[fmt] + 1
    min_exp = BIASED_EXP[fmt][0]
    max_exp = BIASED_EXP[fmt][1]
    max_a_exp = max_exp-(p+5)
    a_exp = random.randint(min_exp, max_a_exp)
    b_exp_nums = max_a_exp - a_exp
    min_b_exp = max_exp - b_exp_nums
    b_exp = random.randint(min_b_exp, max_exp)
    
    exponent_list = [[a_exp, b_exp], [b_exp, a_exp]]
    return exponent_list

def generate_checkerboards(length: int) -> Generator[str, None, None]:
    for zeros_length in range(1, length // 2 + 1):
        # for ones_length in range(1, length // 2 + 1):
        ones_length = zeros_length

        pattern = '0' * zeros_length + '1' * ones_length
        pattern *= length // (zeros_length + ones_length) + 1
        yield pattern[:length]
        yield pattern[::-1][:length]

def generate_leading_and_trailing_zeros(length: int) -> Generator[str, None, None]:
    for zeros_length in range(1, length + 1):
        pattern = '0' * zeros_length + '1' + bin(random.getrandbits(length))[2:]
        yield pattern[:length]
        yield pattern[:length][::-1]

def generate_leading_and_trailing_ones(length: int) -> Generator[str, None, None]:
    for ones_length in range(1, length + 1):
        pattern = '1' * ones_length + '0' + bin(random.getrandbits(length))[2:]
        yield pattern[:length]
        yield pattern[:length][::-1]

def generate_with_k_ones(k: int, length: int, limit: int) -> Generator[str, None, None]:
    for _ in range(limit):
        pattern = random.sample(range(length), k)
        bits = ['0'] * length
        for i in pattern:
            bits[i] = '1'

        yield ''.join(bits)

def generate_with_long_runs(min_run_length: int, length: int) -> Generator[str, None, None]:
    # for start in range(length - min_run_length + 1):
    start = random.randint(0, length - min_run_length)

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

def generate_special_significands(fmt: str) -> List[str]:
    mantissa_length = MANTISSA_BITS[fmt]
    ans = []

    # Leading zeros
    ans.extend(generate_leading_and_trailing_zeros(mantissa_length))

    # Leading ones
    ans.extend(generate_leading_and_trailing_ones(mantissa_length))

    # Small number of 1s
    for k in range(1, mantissa_length // 2 + 1):
        ans.extend(generate_with_k_ones(k, mantissa_length, 1))

    # Small number of 0s
    for k in range(1, mantissa_length // 2 + 1):
        ans.extend(generate_with_k_ones(mantissa_length - k, mantissa_length, 1))

    # Checkerboard patterns
    ans.extend(generate_checkerboards(mantissa_length))

    # Long runs of 1s and 0s
    for run_length in range(1, mantissa_length + 1):
        ans.extend(generate_with_long_runs(run_length, mantissa_length))

    return ans

def main():
    total_tests = 0
    with open("tests/testvectors/B11_tv.txt", "w") as test_f, open("tests/covervectors/B11_cv.txt", "w") as cover_f:
        for fmt in FLOAT_FMTS:
            special_significands = generate_special_significands(fmt)
            for op in [OP_ADD, OP_SUB]:
                random.shuffle(special_significands)
                for a_sig in special_significands:
                    exponent_list = innerTest(fmt)+outerTest(fmt)
                    random.shuffle(exponent_list)
                    exponent_index = 0
                    for b_sig in special_significands:
                        if(a_sig != b_sig):
                            a_sign = random.randint(0,1)
                            b_sign = random.randint(0,1)
                            a_exp = exponent_list[exponent_index][0]
                            b_exp = exponent_list[exponent_index][0]
                            complete_a = decimalComponentsToHex(fmt, a_sign, a_sig, a_exp)
                            complete_b = decimalComponentsToHex(fmt, b_sign, b_sig, b_exp)
                            # print(complete_a)
                            run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00", test_f, cover_f)
                            total_tests+=1
                            exponent_index +=1
                            exponent_index %= len(exponent_list)
    print(total_tests)
            

if __name__ == '__main__':
    main()