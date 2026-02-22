#Lamarr
#B5 Model


import random
from cover_float.reference import run_and_store_test_vector
from cover_float.common.constants import UNBIASED_EXP, OP_CFF, FMT_QUAD, FMT_DOUBLE, FMT_SINGLE, FMT_BF16, FMT_HALF, ROUNDING_MODES, MANTISSA_BITS, ROUNDING_MODES, EXPONENT_BITS, EXPONENT_BIAS


B5_FMTS = [FMT_QUAD, FMT_DOUBLE, FMT_SINGLE, FMT_BF16, FMT_HALF]


def generate_FP(input_e_bitwidth, input_sign, input_exponent, input_mantissa, input_bias):
    exponent = f"{input_exponent + input_bias:0{input_e_bitwidth}b}"
    complete = input_sign + exponent + input_mantissa
    fp_complete = f"{int(complete, 2):032X}"
   
    return fp_complete


def tests_conversion_1_2(lp, hp, rounding_mode, test_f, cover_f):
    hp_m_bits = MANTISSA_BITS[hp]
    hp_e_bits = EXPONENT_BITS[hp]
    hp_e_bias = EXPONENT_BIAS[hp]
    lp_min_exp = UNBIASED_EXP[lp][0] - 1 #Account for subnorms
    lp_m_bits = MANTISSA_BITS[lp]
   
    hp_min_exp = lp_min_exp
   
    if(hp != FMT_SINGLE and lp != FMT_BF16):
        hp_min_exp = lp_min_exp - (lp_m_bits - 1)
       
    input_1_exponent = random.randint(hp_min_exp, lp_min_exp)
    input_2_exponent = random.randint(hp_min_exp, lp_min_exp)
   
    input_1_trailing_0s = (hp_m_bits - lp_m_bits) + (lp_min_exp - input_1_exponent)
    input_2_trailing_0s = (hp_m_bits - lp_m_bits) + (lp_min_exp - input_2_exponent)
   
    input_1_min_mantissa = int('1'+ input_1_trailing_0s * '0', 2)
    input_2_min_mantissa = int('1'+ input_2_trailing_0s * '0', 2)
    max_mantissa = int('1'* hp_m_bits, 2)
   
    input_1_mantissa = format(random.randint(input_1_min_mantissa, max_mantissa), '0b')
    input_2_mantissa = format(random.randint(input_2_min_mantissa, max_mantissa), '0b')
   
    input_value_1 = generate_FP(hp_e_bits, '0', input_1_exponent, input_1_mantissa, hp_e_bias)
    input_value_2 = generate_FP(hp_e_bits, '1', input_2_exponent, input_2_mantissa, hp_e_bias)
   
    run_and_store_test_vector(f"{OP_CFF}_{rounding_mode}_{input_value_1}_{32*'0'}_{32*'0'}_{hp}_{32*'0'}_{lp}_00", test_f, cover_f) #Test 1
    run_and_store_test_vector(f"{OP_CFF}_{rounding_mode}_{input_value_2}_{32*'0'}_{32*'0'}_{hp}_{32*'0'}_{lp}_00", test_f, cover_f) #Test 2
   
# def test_conversion_3(lp_min_exp, hp_min_exp, lp_opcode, hp_opcode, rounding_mode, hp_m_bits, hp_e_bits, ):
   
def convertTests(test_f, cover_f):
    #All conversion tests:
    for i_hp in range(len(B5_FMTS)):
        hp = B5_FMTS[i_hp]
        for i_lp in range(i_hp+1, len(B5_FMTS)):
            lp = B5_FMTS[i_lp]
            for rounding_mode in ROUNDING_MODES:
                tests_conversion_1_2(lp, hp, rounding_mode,test_f, cover_f)#Call tests 1 and 2
               
           
           


def main():
    with open("./tests/testvectors/B5_tv.txt", "w") as test_f, open("./tests/covervectors/B5_cv.txt", "w") as cover_f:
        convertTests(test_f, cover_f)


if __name__ == "__main__":
    main()