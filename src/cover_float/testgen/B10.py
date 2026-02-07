#Lamarr
#B10 Model
#This model tests every possible value for a shift between the input operands
#Definitions)
#    max_exp = the max exponent value (biased, based on precision)
#    min_exp = the min exponent value (biased, based on precision)
#    a = addend 1
#    b = addend 2
#    b_exp = the value of b's exponent (range: [min_exp, max_exp])
#    a_exp = the value of a's exponent (range: [min_exp, max_exp])
#    p = the number of mantissa bits + 1 (based on precision)
#    
#Tests Required:
#    Differences between unbiased exponents must be ->
#1) 1 value smaller than -(p+4)
#2) All values in the range [-(p+4), (p+4)]
#3) 1 value larger than (p+4)
#
# hypothetical vectors generated: 473(3)
#-------- Test # 1 --------
# (1 value smaller than -(p+4))
# a_exp - b_exp < -(p + 4)
#     a_exp generation:
#        max value of a_exp to satisfy this is max_exp - (p + 5)
#        a_exp range must be = [min_exp, max_exp - (p + 5)]
#     b_exp generation:
#        the possible values of b_exp are restricted by a_exp
#        the higher the value of a_exp, the smaller range of b_exp
#        possible # of b_exp values = max_exp - a_exp
#        range for b_exp values = [max_exp - (max_a_exp - a_exp), max_exp]
# all other values defining a and b can be randomly generated
#-------- Test #3 --------
# (1 value larger than (p+4))
# Call Test 1 Function, but swap a and b
#-------- Test #2 --------
# (All values in the range [-(p+4), (p+4)])
#  there will be two processes, one where b_exp is incremented and one where b_exp is decremented
#   where b is incremented)
#    a_exp generation:
#        b_exp must be able to be incrementented p+4 times, starting at a_exp
#        Therefore, a_exp must be limited to [min_exp, max_exp -(p + 4)]
#    b_exp generation:
#        a_exp = b_exp
#    looping:
#        Must loop p + 5 times, because we will start at a_exp = b_exp
#        Generate vectors with a_exp and b_exp
#        Increment b by 1
#  where b is decremented)
#    a_exp generation:
#        b_exp must be able to be decremented p + 3 times, starting at a_exp - 1
#        Therefore, a_exp must be limited to [min_exp + p + 4, max_exp]
#    looping:
#        Must loop p + 4 times
#        Generate vectors with a_exp and b_exp
#        Decrement b by 1


import random
import subprocess


from cover_float.reference import run_and_store_test_vector
from cover_float.common.constants import *




def decimalComponentsToHex(fmt, sign, biased_exp):
    b_sign = f"{sign:01b}"
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
            complete_a = decimalComponentsToHex(fmt, 0, a_exp)
            complete_b = decimalComponentsToHex(fmt, 0, b_exp)
           
            run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00", test_f, cover_f)


            b_exp +=1 #Final statement, increments 1 over
             
        #Decrementing b_exp
       
        a_exp = random.randint(min_exp + (p + 4), max_exp)
        b_exp = a_exp-1
       
        for i in range(0, p+4):
            complete_a = decimalComponentsToHex(fmt, 0, a_exp)
            complete_b = decimalComponentsToHex(fmt, 0, b_exp)
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
       
        complete_a = decimalComponentsToHex(fmt, 0, a_exp)
        complete_b = decimalComponentsToHex(fmt, 0, b_exp)
               
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
   
    # decimalComponentsToHex(FMT_HALF, 0, 25, 976)2000, correct
    # decimalComponentsToHex(FMT_HALF, 0, 19, 256)200, correct
   


if __name__ == "__main__":
    main()

