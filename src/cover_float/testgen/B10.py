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
# hypothetical vectors generated: 473
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
import coverfloat

TEST_VECTOR_WIDTH_HEX  = 144
TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS = (TEST_VECTOR_WIDTH_HEX + 8)

OP_ADD    = "00000010"
ROUND_NEAR_EVEN   = "00"

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

FMTS     = [FMT_HALF, FMT_SINGLE, FMT_DOUBLE, FMT_QUAD, FMT_BF16]
INT_FMTS = [FMT_INT, FMT_UINT, FMT_LONG, FMT_ULONG] #TODO: Do I need to include INT_FMTS?

MANTISSA_BITS = {
    FMT_HALF : 10,
    FMT_SINGLE: 23,
    FMT_DOUBLE: 52,
    FMT_QUAD : 112,
    FMT_BF16 : 7
}

EXPONENT_BITS = {
    FMT_HALF : 5,
    FMT_SINGLE: 8,
    FMT_DOUBLE: 11,
    FMT_QUAD : 15,
    FMT_BF16 : 8
}

BIASED_EXP = { # Range of biased exponents based on precision
    FMT_HALF : [1, 30],
    FMT_SINGLE : [1, 254],
    FMT_DOUBLE: [1, 2046],
    FMT_QUAD: [1, 32766],
    FMT_BF16 : [1, 254]
}

a_mant = 200 #random values because the mantissas don't matter
b_mant = 2000


def decimalComponentsToHex(fmt, sign, biased_exp, mantissa):
    b_sign = f"{sign:01b}"
    b_exponent = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_mantissa = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    b_complete = b_sign + b_exponent + b_mantissa
    h_complete = f"{int(b_complete, 2):032X}"
    return h_complete


def innerTest(f):
    for fmt in FMTS:
        p = MANTISSA_BITS[fmt] + 1
        min_exp = BIASED_EXP[fmt][0]
        max_exp = BIASED_EXP[fmt][1]
        
        #Incrementing b_exp
        
        a_exp = random.randint(min_exp , max_exp - (p+4))
        b_exp = a_exp
        
        for i in range(0, p+5):
            complete_a = decimalComponentsToHex(fmt, 0, a_exp, a_mant)
            complete_b = decimalComponentsToHex(fmt, 0, b_exp, b_mant)
            
            print(coverfloat.reference(f"{OP_ADD}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)

            b_exp +=1 #Final statement, increments 1 over
             
        #Decrementing b_exp
        
        a_exp = random.randint(min_exp + (p + 4), max_exp)
        b_exp = a_exp-1
        
        for i in range(0, p+4):
            complete_a = decimalComponentsToHex(fmt, 0, a_exp, a_mant)
            complete_b = decimalComponentsToHex(fmt, 0, b_exp, b_mant)
            print(coverfloat.reference(f"{OP_ADD}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)

            b_exp -=1 #Final statement, decrements 1 under
    

def outerTest(isTestOne, f):
    for fmt in FMTS:
        p = MANTISSA_BITS[fmt] + 1
        min_exp = BIASED_EXP[fmt][0]
        max_exp = BIASED_EXP[fmt][1]
        max_a_exp = max_exp-(p+5)
        a_exp = random.randint(min_exp, max_a_exp)
        b_exp_nums = max_a_exp - a_exp
        min_b_exp = max_exp - b_exp_nums
        b_exp = random.randint(min_b_exp, max_exp)
        
        complete_a = decimalComponentsToHex(fmt, 0, a_exp, a_mant)
        complete_b = decimalComponentsToHex(fmt, 0, b_exp, b_mant)
                
        if(isTestOne):
            print(coverfloat.reference(f"{OP_ADD}_{ROUND_NEAR_EVEN}_{complete_a}_{complete_b}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)
        else:
            print(coverfloat.reference(f"{OP_ADD}_{ROUND_NEAR_EVEN}_{complete_b}_{complete_a}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)    


def main():
    with open("./tests/testvectors/B10_tv.txt", "w") as f:
        outerTest(True, f) #Test #1
        innerTest(f) #Test #2
        outerTest(False, f) #Test #3
    
    # decimalComponentsToHex(FMT_HALF, 0, 25, 976)2000, correct
    # decimalComponentsToHex(FMT_HALF, 0, 19, 256)200, correct
    

if __name__ == "__main__":
    main()