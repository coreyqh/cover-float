#By: Sisi Wang
#B14.py
#Fuse Multiply Add(FMA)
#B14 -> Multiply-Add: Shift


#This model tests every possible value for a shift between the addends of the multiply-add operation.
#For the difference between the unbiased exponent of the addend and the unbiased exponent of the result of the multiplication, 
#test the following values:
#1.A value smaller than -(2* p + 1)
#2.All the values in the range:[-(2*p +1), (p +1) ]
#3.A value larger than (p + 1)


import random
import subprocess
import coverfloat

TEST_VECTOR_WIDTH_HEX  = 144
TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS = (TEST_VECTOR_WIDTH_HEX + 8)


# Dictionaries
OP_ADD = "00000010"
OP_FMA    = "00000050"
OP_FMADD  = "00000051"
OP_FMSUB  = "00000052"
OP_FNMADD = "00000053"
OP_FNMSUB = "00000054"
ROUND_NEAR_EVEN = "00"

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
INT_FMTS = [FMT_INT, FMT_UINT, FMT_LONG, FMT_ULONG] 

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

#safe operating range for exponents
BIASED_EXP = { # Range of biased exponents based on precision
    FMT_HALF : [1, 30], 
    FMT_SINGLE : [1, 254], # ex: 0 is subnormal, 255 is Inf/NaN
    FMT_DOUBLE: [1, 2046],
    FMT_QUAD: [1, 32766],
    FMT_BF16 : [1, 254]
}

OPS = [OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]


def decimalComponentsToHex(fmt, sign, biased_exp, mantissa):
    b_sign = f"{sign:01b}"
    b_exponent = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_mantissa = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    b_complete = b_sign + b_exponent + b_mantissa
    h_complete = f"{int(b_complete, 2):032X}"
    return h_complete





def generate_b14_tests(f,fmt):
    p = MANTISSA_BITS[fmt] + 1 #defines the precision we are working with
    bias = (1 << (EXPONENT_BITS[fmt] - 1)) - 1 #calculates the correct bias depending on fmt
    min_exp = BIASED_EXP[fmt][0]
    max_exp = BIASED_EXP[fmt][1]


    # Define Format-Specific Shift Limits
    # This defines the sweep range [ -limit, +limit ]
    # We want to cover the full range of (ExpA + ExpB) - ExpC
    SHIFT_LIMITS = {
        FMT_HALF   : 60,    # Max exp diff is ~30. We use 60 
        FMT_BF16   : 260,   # Max exp diff is ~255. We use 260 
        FMT_SINGLE : 300,   # Max exp diff is ~255. We use 300 
        FMT_DOUBLE : 2200,  # Max exp diff is ~2047. We use 2200 
        FMT_QUAD   : 5000   # Max exp diff is ~32000. We cap at 5000 to save time.
    }
    
    # Get the limit (default to 500 if format missing)
    limit = SHIFT_LIMITS.get(fmt, 500)
    
    start_shift = -limit 
    end_shift   = limit
    
    for target_shift in range(start_shift, end_shift + 1):
        
        for op in OPS:
        # Randomize & generate 50 variations per shift
            for _ in range(50): 
            
                ##Part 1: Randomize a and b exponents (and make sure their product is valid)
                #a:
                safe_margin = int(max_exp / 4) #safe margin defined to keep 'a' somewhat central to avoid immediate overflows
                exp_a = random.randint(min_exp + safe_margin, max_exp - safe_margin)
                #b:
                low_bound = max(min_exp, bias - 75)
                high_bound = min(max_exp, bias + 75)
                exp_b = random.randint(low_bound, high_bound)  # Pick 'b' near the bias (so product exponent is close to exp_a) or random. This is simplified; we might want more range here.
                

                ##Part 2: Calculate the addends
                # Calculate product exponent:Exp_Prod = Exp_A + Exp_B - Bias
                exp_prod = exp_a + exp_b - bias
                
                #Calculate required Exp_C to hit the Target Shift: target_shift = Exp_C - Exp_Prod
                exp_c = target_shift + exp_prod
                
                #Quick validity check -> If the calculated exp_c is invalid (too big/small), skip this variation
                if exp_c < min_exp or exp_c > max_exp:
                    continue

                ##Part 3: Generate mantissa componenets + Assemble >:3
                # Create random signs (0 or 1)
                sign_a = random.randint(0, 1)
                sign_b = random.randint(0, 1)
                sign_c = random.randint(0, 1)

                #Randomize mantissas -> Uses random bits to trigger different carry/rounding paths.
                mant_a = random.getrandbits(MANTISSA_BITS[fmt])
                mant_b = random.getrandbits(MANTISSA_BITS[fmt])
                mant_c = random.getrandbits(MANTISSA_BITS[fmt])
            
                #Converts the components created above to hex
                hex_a = decimalComponentsToHex(fmt, sign_a, exp_a, mant_a)
                hex_b = decimalComponentsToHex(fmt, sign_b, exp_b, mant_b)
                hex_c = decimalComponentsToHex(fmt, sign_c, exp_c, mant_c)  

                print(coverfloat.reference(f"{op}_{ROUND_NEAR_EVEN}_{hex_a}_{hex_b}_{hex_c}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)

def main():
    with open("./tests/testvectors/B14_tv.txt", "w") as f:
        for fmt in FMTS:
            generate_b14_tests(f,fmt)


if __name__ == "__main__":
    main()









