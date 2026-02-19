#Lamarr
#B10 Model
#This model tests every possible value for a shift between the input operands

import random
import subprocess


from cover_float.reference import run_and_store_test_vector
from cover_float.common.constants import FLOAT_FMTS, EXPONENT_BITS, MANTISSA_BITS, BIASED_EXP, ROUND_NEAR_EVEN, OP_ADD, OP_SUB




def decimalComponentsToHex(fmt, biased_exp):
    b_sign = f"{random.randint(0,1)}"
    b_exponent = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_mantissa = f"{random.getrandbits(MANTISSA_BITS[fmt]):0{MANTISSA_BITS[fmt]}b}"
    b_complete = b_sign + b_exponent + b_mantissa
    h_complete = f"{int(b_complete, 2):032X}"
    return h_complete




def innerTest(test_f, cover_f, op):
    for fmt in FLOAT_FMTS:
        p = MANTISSA_BITS[fmt] + 1
        min_exp = BIASED_EXP[fmt][0]
        max_exp = BIASED_EXP[fmt][1]
       
        #Incrementing b_exp
       
        a_exp = random.randint(min_exp , max_exp - (p+4))
        b_exp = a_exp
       
        for i in range(0, p+5):
            complete_a = decimalComponentsToHex(fmt, a_exp)
            complete_b = decimalComponentsToHex(fmt, b_exp)
           
            run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00", test_f, cover_f)


            b_exp +=1 #Final statement, increments 1 over
             
        #Decrementing b_exp
       
        a_exp = random.randint(min_exp + (p + 4), max_exp)
        b_exp = a_exp-1
       
        for i in range(0, p+4):
            complete_a = decimalComponentsToHex(fmt, a_exp)
            complete_b = decimalComponentsToHex(fmt, b_exp)
            run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00", test_f, cover_f)


            b_exp -=1 #Final statement, decrements 1 under
   


def outerTest(isTestOne, test_f, cover_f, op):
    for fmt in FLOAT_FMTS:
        p = MANTISSA_BITS[fmt] + 1
        min_exp = BIASED_EXP[fmt][0]
        max_exp = BIASED_EXP[fmt][1]
        max_a_exp = max_exp-(p+5)
        a_exp = random.randint(min_exp, max_a_exp)
        b_exp_nums = max_a_exp - a_exp
        min_b_exp = max_exp - b_exp_nums
        b_exp = random.randint(min_b_exp, max_exp)
       
        complete_a = decimalComponentsToHex(fmt, a_exp)
        complete_b = decimalComponentsToHex(fmt, b_exp)
               
        if(isTestOne):
            run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00", test_f, cover_f)
        else:
            run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{complete_b}_{complete_a}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00", test_f, cover_f)




def main():
    with open("./tests/testvectors/B10_tv.txt", "w") as test_f, open("./tests/covervectors/B10_cv.txt", "w") as cover_f:
        for op in [OP_ADD, OP_SUB]:
            outerTest(True, test_f, cover_f, op) #Test #1
            innerTest(test_f, cover_f, op) #Test #2
            outerTest(False, test_f, cover_f, op) #Test #3
   


if __name__ == "__main__":
    main()