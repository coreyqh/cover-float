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
    lp_sn_exp = UNBIASED_EXP[lp][0] - 1 #Account for subnorms
    lp_m_bits = MANTISSA_BITS[lp]

    hp_max_exp = lp_sn_exp - 1 #double check this
    hp_min_exp = lp_sn_exp - lp_m_bits
   
    if(hp == FMT_SINGLE and lp == FMT_BF16): #Different case for FP_32 and BF_16
        hp_min_exp = lp_sn_exp
        hp_max_exp = lp_sn_exp

    input_1_exponent = random.randint(hp_min_exp, hp_max_exp)
    input_2_exponent = random.randint(hp_min_exp, hp_max_exp)
   
    max_mantissa = int('1'* hp_m_bits, 2)
   
    input_1_mantissa = format(random.randint(0, max_mantissa), '0b')
    input_2_mantissa = format(random.randint(0, max_mantissa), '0b')
   
    input_value_1 = generate_FP(hp_e_bits, '0', input_1_exponent, input_1_mantissa, hp_e_bias)
    input_value_2 = generate_FP(hp_e_bits, '1', input_2_exponent, input_2_mantissa, hp_e_bias)
   
    run_and_store_test_vector(f"{OP_CFF}_{rounding_mode}_{input_value_1}_{32*'0'}_{32*'0'}_{hp}_{32*'0'}_{lp}_00", test_f, cover_f) #Test 1
    run_and_store_test_vector(f"{OP_CFF}_{rounding_mode}_{input_value_2}_{32*'0'}_{32*'0'}_{hp}_{32*'0'}_{lp}_00", test_f, cover_f) #Test 2
   
# def genTestVectors3_4():
    
def tests_conversion_3_4(lp, hp, rounding_mode, test_f, cover_f):
    hp_m_bits = MANTISSA_BITS[hp]
    hp_e_bits = EXPONENT_BITS[hp]
    hp_e_bias = EXPONENT_BIAS[hp]
    lp_sn_exp = UNBIASED_EXP[lp][0] - 1 #Account for subnorms
    lp_m_bits = MANTISSA_BITS[lp]
    
    hp_sn_lp_exp = lp_sn_exp - lp_m_bits
    
    #MinSN + 1 ulp, MinSN + 2 ulp, MinSN + 3 ulp
    for i in range(0, int('11', 2)): # Iterate over differt round and sticky bits
        hp_exp = hp_sn_lp_exp
        rs = format(i, '0b')#round and sticky bit
        remaining_mantissa_bits = hp_m_bits - 2 #find out what part of the mantissa bits left need to be randomized
        
        complete_binary_1 = rs + format(random.randint(0, int('1'*remaining_mantissa_bits, 2)), '0b')#randomize the rest of the number
        complete_binary_2 = rs + format(random.randint(0, int('1'*remaining_mantissa_bits, 2)), '0b')
        
        generate_FP(hp_e_bits, '0', hp_exp, complete_binary_1, hp_e_bias)
        generate_FP(hp_e_bits, '1', hp_exp, complete_binary_2, hp_e_bias) 
    
    #MinSN - 1 ulp, MinSN - 2 ulp
    for i in range(int('10',2), int('11',2)):
        hp_exp = hp_sn_lp_exp - 1
        s = format(i, '0b')#sticky bit
        remaining_mantissa_bits = hp_m_bits - 1
        
        complete_binary_1 = s + format(random.randint(0, int('1'*remaining_mantissa_bits, 2)), '0b')#randomize the rest of the number
        complete_binary_2 = s + format(random.randint(0, int('1'*remaining_mantissa_bits, 2)), '0b')
        
        input_1 = generate_FP(hp_e_bits, '0', hp_exp, complete_binary_1, hp_e_bias)
        input_2 = generate_FP(hp_e_bits, '1', hp_exp, complete_binary_2, hp_e_bias)
        
        run_and_store_test_vector(f"{OP_CFF}_{rounding_mode}_{input_1}_{32*'0'}_{32*'0'}_{hp}_{32*'0'}_{lp}_00", test_f, cover_f) #Test 1
        run_and_store_test_vector(f"{OP_CFF}_{rounding_mode}_{input_2}_{32*'0'}_{32*'0'}_{hp}_{32*'0'}_{lp}_00", test_f, cover_f) #Test 2
        
    #MinSN - 3 ulp
    hp_exp = hp_sn_lp_exp - 2
    s = format(i, '0b')#sticky bit
    remaining_mantissa_bits = hp_m_bits

    complete_binary_1 = format(random.randint(0, int('1'*hp_m_bits, 2)), '0b')#randomize the rest of the number
    complete_binary_2 = format(random.randint(0, int('1'*hp_m_bits, 2)), '0b')

    generate_FP(hp_e_bits, '0', hp_exp, complete_binary_1, hp_e_bias)
    generate_FP(hp_e_bits, '1', hp_exp, complete_binary_2, hp_e_bias)
    
    
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