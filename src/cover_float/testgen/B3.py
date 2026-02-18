import random
import cover_float.common as common
from cover_float.reference import run_test_vector, store_cover_vector

SRC1_OPS = [common.OP_SQRT]

SRC2_OPS = [common.OP_ADD,
            common.OP_SUB,
            common.OP_MUL,
            common.OP_DIV]
            # common.OP_REM]

SRC3_OPS = [
    #common.OP_FMA,
            common.OP_FMADD,
            common.OP_FMSUB,
            common.OP_FNMADD,
            common.OP_FNMSUB]

# OP_QC     = "000000B0"
# OP_FEQ    = "000000B1"
# OP_SC     = "000000C0"
# OP_FLT    = "000000C1"
# OP_FLE    = "000000C2"
# OP_CLASS  = "000000D0"
# OP_MIN    = "000000E0"
# OP_MAX    = "000000F0"
# OP_CSN    = "00000100"
# OP_FSGNJ  = "00000101"
# OP_FSGNJN = "00000102"
# OP_FSGNJX = "00000103"

FMTS     = [common.FMT_SINGLE, common.FMT_DOUBLE, common.FMT_HALF, common.FMT_BF16, common.FMT_QUAD]
INT_FMTS = [common.FMT_INT, common.FMT_UINT, common.FMT_LONG, common.FMT_ULONG]


def generate_float(sign: int, exponent: int, mantissa: int, fmt: str) -> int:
    exponent += common.EXPONENT_BIASES[fmt]
    return (sign << (common.MANTISSA_BITS[fmt] + common.EXPONENT_BITS[fmt])) | (exponent << common.MANTISSA_BITS[fmt]) | mantissa

def generate_random_float(exponent: int, fmt: str) -> int:
    sign = random.randint(0, 1)
    # sign = 0
    mantissa = random.randint(0, (1 << common.MANTISSA_BITS[fmt]) - 1)
    # Add in the exponent bias for single-precision (127)
    float32 = generate_float(sign, exponent, mantissa, fmt)

    return float32

def get_significand_from_float(float_: int, fmt: str) -> int:
    mask = (1 << common.MANTISSA_BITS[fmt]) - 1
    return float_ & mask | (1 << common.MANTISSA_BITS[fmt])

def generate_test_vector(op, in1, in2, in3, fmt1, fmt2, rnd_mode="00"):
    zero_padding = '0' * 32
    return f"{op}_{rnd_mode}_{in1:032x}_{in2:032x}_{in3:032x}_{fmt1}_{zero_padding}_{fmt2}_00\n"


def extract_rounding_info(cover_vector):
    fields = cover_vector.split('_')
    sgn = fields[-3]
    exp = int(fields[-2], 16)
    result_fmt = fields[-5]
    # Sketchy convert to signed int
    signed_exp = int.from_bytes(exp.to_bytes(4, 'little', signed=False), 'little', signed=True)
    
    interm_significand = int('1' + fields[-1], 16)
    # if result_fmt == common.FMT_QUAD:
    #     interm_significand = bin(interm_significand)[2:][16:]
    # else:
    #     interm_significand = bin(interm_significand)[2:][2:]
    interm_significand = bin(interm_significand)[2:][1:]
    if signed_exp < 0:
        interm_significand = '0' * (-signed_exp + 1) + interm_significand

    # breakpoint()

    mantissa_length = common.MANTISSA_BITS[result_fmt]

    # if result_fmt == common.FMT_BF16:
    #     return { 'Sign': 0, 'LSB': 0, 'Guard': 0, 'Sticky': 0 }
    if interm_significand == '0':
        # breakpoint()
        return { 'Sign': 2, 'LSB': 0, 'Guard': 0, 'Sticky': 0 }
    
    # if (signed_exp > 0 and int(fields[-1][0], 16) & 0x4 == 0) and not (result_fmt == common.FMT_QUAD and fields[-1][:4] == '0001'):
    #     breakpoint()

    lsb = interm_significand[mantissa_length - 1]
    guard = interm_significand[mantissa_length]
    sticky = interm_significand[mantissa_length + 1:]

    # if result_fmt == common.FMT_QUAD:
    #     quad_offset = 15
    #     lsb = interm_significand[mantissa_length - 1 + quad_offset]
    #     guard = interm_significand[mantissa_length + quad_offset]
    #     sticky = interm_significand[mantissa_length + 1 + quad_offset:]


    f32_mantissa = int(fields[-1][:16], 16)

    # breakpoint()

    # print(bin(f32_mantissa & 0x1ff))
    # print(sticky)

    return {
        'Sign': int(sgn),
        'LSB': int(lsb),
        'Guard': int(guard),
        'Sticky': 1 if any(x == '1' for x in sticky) else 0,
    }

def write_fma_tests(test_f, cover_f, fmt):
    FMA_OPS = [
        common.OP_FMADD,
        common.OP_FMSUB,
        common.OP_FNMADD,
        common.OP_FNMSUB,
    ]

    targets = [
        {
            'Sign': (x & 1),
            'LSB': (x & 2) >> 1,
            'Guard': (x & 4) >> 2,
            'Sticky': 0,
        }
        for x in range(8)
    ]

    for op in FMA_OPS:
        for mode in common.ROUNDING_MODES:
            to_cover = targets[:]

            for _ in range(100):
                """
                    How does FMA actually work on softfloat? (and why we are not using the reference 
                    model to do our math)
                    
                    Softfloat is going to crush extra bits into one with the shiftJam logic, and in 
                    the f32 case, softfloat's rounding function takes a uint_fast32_t as input for
                    the significand. This means that it rounds based off of ~9 extra bits instead
                    of all of the generated sticky bits (so we cannot get preaddition results
                    with an OP_FMADD x, y, 0 call). 

                    The following is a calculation from s_mulAddF32.c:

                        sigC = (sigC | 0x00800000)<<6;

                        ...

                        sig64Z =
                            sigProd
                                + softfloat_shiftRightJam64(
                                    (uint_fast64_t) sigC<<32, expDiff );
                        sigZ = softfloat_shortShiftRightJam64( sig64Z, 32 );
                    
                    sig64Z is a uint_fast64_t, while sigZ is a uint_fast32_t. SigZ is the final answer
                    but what we want is sig64Z. The meaning of Jam is that bits shifted out of the integer
                    are "jammed" into a 1. Thus, we just need a faithful calculation of sigProd.

                    So, how is sigProd calculated?

                        sigA = (sigA | 0x00800000)<<7;
                        sigB = (sigB | 0x00800000)<<7;
                        sigProd = (uint_fast64_t) sigA * sigB;
                        if ( sigProd < UINT64_C( 0x2000000000000000 ) ) {
                            --expProd;
                            sigProd <<= 1;
                        }

                    And expProd? (This is off by one because softfloat rounding is weird)

                        expProd = expA + expB - 0x7E // 0x7e = 126
                """

                signA = random.randint(0, 1)
                signB = random.randint(0, 1)

                sigA_initial = random.randint(0, (1 << common.MANTISSA_BITS[fmt]) - 1)
                sigB_initial = random.randint(0, (1 << common.MANTISSA_BITS[fmt]) - 1)
                expA = random.randint(-10, 10) + common.EXPONENT_BIASES[fmt]
                expB = random.randint(-10, 10) + common.EXPONENT_BIASES[fmt]

                if fmt == common.FMT_HALF:
                    # Just be careful that we don't generate things that need
                    # to add a number that we don't have the exponents to add
                    expA = random.randint(-1, 6) + common.EXPONENT_BIASES[fmt]
                    expB = random.randint(-1, 6) + common.EXPONENT_BIASES[fmt]

                # Put in the leading one
                sigA = (sigA_initial | (1 << common.MANTISSA_BITS[fmt]))
                sigB = (sigB_initial | (1 << common.MANTISSA_BITS[fmt]))
                
                # Actually Multiply
                sigProd = sigA * sigB
                signProd = signA ^ signB # zero iff both are the same
                expProd = expA + expB - common.EXPONENT_BIASES[fmt] + 1
                
                # Correct for the actual operation
                if op == common.OP_FNMADD or op == common.OP_FNMSUB:
                    signProd ^= 1 # These ops induce a sign flip
                
                # Now we ensure that our leading one is in the correct bit, and the 
                # product exponent is correct
                if sigProd < (1 << (common.MANTISSA_BITS[fmt] * 2 + 1)):
                    sigProd <<= 1
                    expProd -= 1
                
                # print("expProd (python):", expProd)

                # The leading one should be in bit MANTISSA_BIT * 2 + 2, so
                # bits MANTISSA_BIT * 2 + 1 --> MANTISSA_BIT + 2 (inclusive) are mantissa
                # Thus, G = MANTISSA_BIT + 1, STICKY = MANTISSA_BIT --> 1
                mask = 2 ** (common.MANTISSA_BITS[fmt] + 1) - 1
                rounding_bits = sigProd & mask
                sticky_bits = rounding_bits & (mask >> 1)
                not_sticky = sigProd & (~mask)

                # Sticky bits should be aligned to already, so
                signC = signProd
                sigC_initial = 2 ** common.MANTISSA_BITS[fmt] - sticky_bits
                sigC = sigC_initial | (1 << common.MANTISSA_BITS[fmt])

                # Sign Flip if it is a subtraction op
                if op == common.OP_FMSUB or op == common.OP_FNMADD:
                    signC ^= 1

                # Figure out alignment
                expC = expProd - common.MANTISSA_BITS[fmt] - 1 
                expDiff = expProd - expC

                # Align sigC to correct bits of sigProd, the shifts are a no-op but
                # they are there for correctness
                sigZ64 = sigProd + ((sigC << (common.MANTISSA_BITS[fmt] + 1)) >> expDiff) 
                # sigZ64 = sigProd + sigC

                # In some cases, especially in lower precision formats (i.e. bf16 and half), 
                # we get an "overflow" here (i.e. we move up an exponent and have to shift)
                # This means we can accidentally cause a shift of guard into the stickt bit
                # which we do not guarentee to be zero, so we check that here
                if len(bin(sigZ64)) > len(bin(sigProd)):
                    continue

                # Get new rounding info, if we want to log it
                new_rounding = sigZ64 & mask
                new_sticky = new_rounding & (mask >> 1)

                in1 = generate_float(signA, expA - common.EXPONENT_BIASES[fmt], sigA_initial, fmt)
                in2 = generate_float(signB, expB - common.EXPONENT_BIASES[fmt], sigB_initial, fmt)
                in3 = generate_float(signC, expC - common.EXPONENT_BIASES[fmt], sigC_initial, fmt)

                negIn3 = generate_float(signC ^ 1, expC - common.EXPONENT_BIASES[fmt], sticky_bits, fmt)

                tv = generate_test_vector(op, in1, in2, in3, fmt, fmt, mode)
                fake_tv = generate_test_vector(op, in1, in2, 0, fmt, fmt, mode)
                fake_tv_2 = generate_test_vector(common.OP_MUL, in1, in2, 0, fmt, fmt, mode)
                fake_tv_3 = generate_test_vector(op, in1, in2, negIn3, fmt, fmt, mode)
                result = run_test_vector(tv)
                fake_result = run_test_vector(fake_tv)
                fake_result_2 = run_test_vector(fake_tv_2)
                fake_result_3 = run_test_vector(fake_tv_3)

                sig_1 = bin(int(fake_result.split('_')[-1], 16))[2:]
                sig_2 = bin(int(fake_result_2.split('_')[-1], 16))[2:]
                sig_3 = bin(int('1' + fake_result_3.split('_')[-1], 16))[2:][1:]

                expected_sig3 = bin(sigProd - ((1 << common.MANTISSA_BITS[fmt]) + sticky_bits))[2:]
                expected_sig3 = expected_sig3[1:]
                if not (sig_3.startswith(expected_sig3) or (fmt == common.FMT_QUAD and expected_sig3.startswith(sig_3))):
                    breakpoint()


                shiftAmt = len(bin(sigProd)[2:]) - 1 - common.MANTISSA_BITS[fmt]
                shiftedSigProd = sigProd >> shiftAmt
                shiftedSigProd &= (1 << common.MANTISSA_BITS[fmt]) - 1
                in4 = generate_float(signA ^ signB ^ 1, expProd - common.EXPONENT_BIASES[fmt], shiftedSigProd, fmt)
                tv_4 = generate_test_vector(common.OP_FMADD, in1, in2, in4, fmt, fmt, mode)
                out_4 = run_test_vector(tv_4)

                sig_4 = bin(int('1' + out_4.split('_')[-1], 16))[2:][1:]
                first_digit = int(out_4.split('_')[-1][0], 16)
                expected_sig4 = sigProd - (((1 << common.MANTISSA_BITS[fmt]) + shiftedSigProd) << shiftAmt)

                if fmt != common.FMT_QUAD:
                    is_subnormal = int(out_4.split('_')[-2], 16) <= 0
                    if not sig_4.startswith(bin(expected_sig4)[2:][1:]) and not is_subnormal: #or (is_subnormal and not sig_4.startswith(bin(expected_sig4)[2:].zfill(common.MANTISSA_BITS[fmt]-1))): # or (first_digit & 0b1100 != 0b0100 and expected_sig4 != 0 and not is_subnormal):
                        breakpoint()

                # out exp should be exp prod
                result_float = int(result.split('_')[-6], 16)
                result_float >>= common.MANTISSA_BITS[fmt]
                result_float &= (1 << common.EXPONENT_BITS[fmt]) - 1

                inter_exp = int(result.split('_')[-2], 16)

                if expProd != inter_exp and not (fmt == common.FMT_BF16 and abs(expProd - inter_exp) <= 1):
                    breakpoint()

                rounding = extract_rounding_info(result)

                if rounding['Sticky'] != 0:
                    print("FMA Sticky Bit Generation Failed! This should not happen, please investigate")
                    print(f"\tInputs: signA={signA}, sigA={sigA:#x}, expA={expA}, signB={signB}, sigB={sigB:#x}, expB={expB}, fmt={fmt}, op={op}")

                if rounding in to_cover:
                    to_cover.remove(rounding)
                    print(result[:common.TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS], file=test_f)
                    print(result, file=cover_f)

                    # This means were done
                    if len(to_cover) == 0:
                        break
            else:
                # This catches a for loop that does not break, i.e. we don't hit every goal
                # if fmt != common.FMT_BF16: # We have no rounding info extraction on BF16
                print(fmt, mode, to_cover)
            
def write_add_sub_tests(test_f, cover_f, fmt):
    ops = [
        common.OP_ADD,
        common.OP_SUB,
    ]

    targets = [
        {
            'Sign': (x & 1),
            'LSB': (x & 2) >> 1,
            'Guard': (x & 4) >> 2,
            'Sticky': 0,
        }
        for x in range(8)
    ]

    for op in ops:
        for mode in common.ROUNDING_MODES:
            goals = targets[:]
            for _ in range(100):
                target = random.choice(targets)

                # Generate a random float for A
                signA = target['Sign']
                
                # If the MSB of sigA_intial is 0, it prevents rounding up to another exponent
                sigA_initial = random.randint(0, (1 << (common.MANTISSA_BITS[fmt] - 1)) - 1)

                sigA = sigA_initial | (1 << common.MANTISSA_BITS[fmt])
                expA = random.randint(-10, 14) # + common.EXPONENT_BIASES[fmt]

                # How can we get rounding bits to be what we want? 
                # For add and sub, unfortunately, there is no way to get a lot of manipulation, like we could with fma

                # We will misalign them by 2 bits
                expB = expA - 2

                last_digits = ((target['LSB'] ^ (sigA_initial & 1)) << 2) | (target['Guard'] << 1) | (target['Sticky'])

                sigB_initial = (random.randint(1, (1 << common.MANTISSA_BITS[fmt]) - 1) & (~0b111)) + last_digits
                sigB = sigB_initial | (1 << common.MANTISSA_BITS[fmt])
                signB = signA if op == common.OP_ADD else signA ^ 1

                A = generate_float(signA, expA, sigA_initial, fmt)
                B = generate_float(signB, expB, sigB_initial, fmt)

                tv = generate_test_vector(op, A, B, 0, fmt, fmt, mode)
                result = run_test_vector(tv)

                info = extract_rounding_info(result)
                if info in goals:
                    goals.remove(info)
                    store_cover_vector(result, test_f, cover_f)

                    if len(goals) == 0:
                        # break
                        pass
                if target != info:
                    breakpoint()
            else:
                if goals:
                    print(f"write_add_sub_tests failed: cases remaining {goals}")
                    breakpoint()

def write_mul_tests(test_f, cover_f, fmt: str):
    targets = [
        {
            'Sign': (x & 1),
            'LSB': (x & 2) >> 1,
            'Guard': (x & 4) >> 2,
            'Sticky': 0,
        }
        for x in range(8)
    ]

    for mode in common.ROUNDING_MODES:
        """
        We care about setting the last two bits as a result of our multiplication. Perhaps the simplest way
        is to use random chance. That is, we generate things that in theory multiply to the product
        length that we want, and just use random mantissas.
        """

        goals = targets[:]

        for _ in range(100):
            # a_exp_length + b_exp_length = mantissa_length + 1
            # The idea here is that we multiply and get a product significand
            # with length mantissa_length + 1
            a_exp_length = random.randint(3, common.MANTISSA_BITS[fmt] - 2)
            b_exp_length = common.MANTISSA_BITS[fmt] + 1 - a_exp_length 

            # Generate significands
            sig_a_initial = random.randint(1, (1 << a_exp_length) - 1)
            sig_b_initial = random.randint(1, (1 << b_exp_length) - 1)

            # Align them
            sig_a_initial <<= common.MANTISSA_BITS[fmt] - a_exp_length
            sig_b_initial <<= common.MANTISSA_BITS[fmt] - b_exp_length

            # Randomize the rest, and don't overflow
            a_sign = random.randint(0, 1)
            b_sign = random.randint(0, 1)
            a_exp = random.randint(-common.EXPONENT_BIASES[fmt] // 2 + 1, common.EXPONENT_BIASES[fmt] // 2 - 1)
            b_exp = random.randint(-common.EXPONENT_BIASES[fmt] // 2 + 1, common.EXPONENT_BIASES[fmt] // 2 - 1)

            # Run everything
            a = generate_float(a_sign, a_exp, sig_a_initial, fmt)
            b = generate_float(b_sign, b_exp, sig_b_initial, fmt)

            tv = generate_test_vector(common.OP_MUL, a, b, 0, fmt, fmt, mode)
            result = run_test_vector(tv)
            info = extract_rounding_info(result)

            if info in goals:
                goals.remove(info)
                store_cover_vector(result, test_f, cover_f)

                if len(goals) == 0:
                    break
        else:
            print("Failed to generate mul cover_vectors for fmt={fmt}, mode={mode}. Remaining cases {goals}")

def write_sqrt_tests(test_f, cover_f, fmt: str):
    """
    SQRT is fun. LSB  = 1 and Guard = 1 is impossible. We know this because
    consider squaring a number with guard = 1 and an m bit mantissa
    then we have (1.(mantissa)1)**2 = (1 + (mantissa) + 2**(-p-1)) ** 2
    The mantissa has least power 2**(-p), so in the resulting expression there must be
    a powere of 2 ** (-2p - 2), and unfortunately, we cannot represent this. Similar logic
    means that LSB = 1, is also impossible.

    This means that the possible cases are LSB = 0, Guard = 0, Sticky = 0 :(

    Sign is, of course, always zero in these. 

    Note that there are no subnormal tricks here because sqrt halves the exponent and we 
    would be at expmin in a subnorm.
    """

    targets = [
        {
            'Sign': 0,
            'LSB': 0,
            'Guard': 0,
            'Sticky': 0,
        }
    ]

    for mode in common.ROUNDING_MODES:
        # Our life is very easy, we just need a random number filled half way with bits
        usable_bits = common.MANTISSA_BITS[fmt] // 2 - 1
        mantissa = random.randint(1, (1 << (usable_bits)) - 1) 
        mantissa = mantissa << (common.MANTISSA_BITS[fmt] - usable_bits)
        mantissa |= (1 << common.MANTISSA_BITS[fmt])

        # Just something that can be doubled
        exp = random.randint(3, common.EXPONENT_BIASES[fmt] - 3) - (common.EXPONENT_BIASES[fmt] // 2)
        
        # Square the mantissa
        squared_mantissa = mantissa * mantissa
        squared_exp = exp * 2 + 1

        # Align bits correctly (see fma)
        if squared_mantissa < (1 << (common.MANTISSA_BITS[fmt] * 2 + 1)):
            squared_mantissa <<= 1
            squared_exp -= 1

        # Put bits where they are supposed to be
        squared_mantissa >>= common.MANTISSA_BITS[fmt] + 1

        mask = (1 << common.MANTISSA_BITS[fmt]) - 1
        float_ = generate_float(0, squared_exp, squared_mantissa & mask, fmt)
        tv = generate_test_vector(common.OP_SQRT, float_, 0, 0, fmt, fmt, mode)
        
        result = run_test_vector(tv)
        info = extract_rounding_info(result)

        if info not in targets:
            print(f"sqrt generation sticky bit generation failed, please investigate: mantissa={mantissa:x}, exp={exp}")

            float_2 = generate_float(0, exp, mantissa & mask, fmt)
            tv_mul = generate_test_vector(common.OP_MUL, float_2, float_2, 0, fmt, fmt)
            result_mul = run_test_vector(tv_mul)
            gen_square = int(result_mul.split('_')[-6], 16)

            if float_ != gen_square:
                print(f"sqrt float should have been: {gen_square:x}, was {float_:x}")
                return
        else:
            store_cover_vector(result, test_f, cover_f)

def write_div_tests(test_f, cover_f, fmt: str):
    """
    We can generate guard = 1, sticky = 0, unlike square root, but the machinery is going to be 
    very specific. When sticky = 0, we have an exact result. This means that the given quotient
    has a terminating binary expansion. This happens if the given denominator only has a factor
    of two when it is in lowest terms. Let S1 be the first significand, S2 be the second significand,
    p1 and p2 be the powers of their respective floats, and m be the number of mantissa bits. Then
    our quotient is
            S1 * 2**(p1 + m)
            ________________
            S2 * 2**(p1 + m)
    The powers of two cancel, so what must happen for S1 and S2 is that the non-2 factors of each 
    cancel. When S1 and S2 go into lowest terms, we have K / 2^p where K is any odd prime factors
    not canceled out and p is an integer. We canceled factors so K < S1. Thus, the binary representation
    of K must have as many or fewer digits than S1. The meaning of this is that guard = 1 is impossible
    for a normalized generated significand. Thus, guard = 1, sticky = 0 can only be accomplished 
    with a subnorm result. Similarly, K and S1 are only the same length when S1 = K (as we can only
    cancel factors of two or greater), so for lsb = 1 cases, either we need to use trivial significands
    or also use subnorms. 
    """

    targets = [
        {
            'Sign': (x & 1),
            'LSB': (x & 2) >> 1,
            'Guard': (x & 4) >> 2,
            'Sticky': 0,
        }
        for x in range(8)
    ]

    for mode in common.ROUNDING_MODES:
        for target in targets:
            # Generate the subnormal significand that we want to get
            target_subnorm = (random.randint(1, (1 << (common.MANTISSA_BITS[fmt] // 2)) - 1) << 2) | (target['LSB'] << 1) | target['Guard']
            K = target_subnorm
            odd_factors = random.randint(1, (1 << (common.MANTISSA_BITS[fmt] // 3)) - 1)

            sig1_mant = target_subnorm * odd_factors
            sig2_mant = odd_factors

            # Align each of them to have a leading one in bit common.MANTISSA_BITS[fmt]
            sig1_msb = len(bin(sig1_mant)[2:])
            sig1_shift = (common.MANTISSA_BITS[fmt] - sig1_msb + 1)
            sig1 = sig1_mant << sig1_shift
            sign1 = random.randint(0, 1)

            sig2_msb = len(bin(sig2_mant)[2:])
            sig2_shift = common.MANTISSA_BITS[fmt] - sig2_msb + 1
            sig2 = sig2_mant << sig2_shift
            sign2 = sign1 ^ target['Sign']

            # exp1 = random.randint(-common.EXPONENT_BIASES[fmt] // 2, -2)
            exp1 = random.randint(-common.EXPONENT_BIASES[fmt] + 1, -common.MANTISSA_BITS[fmt] + 1)

            # Mirroring soft_float calculation
            # sig1_64 = sig1 << (31 if sig1 < sig2 else 30)
            sig1_64 = sig1 << (common.MANTISSA_BITS[fmt] + 1 if sig1 < sig2 else common.MANTISSA_BITS[fmt])
            sig_quotient = (sig1_64) // sig2

            if sig_quotient * sig2 != sig1_64:
                print(f"Failure to generate exact division result, please investigate: K={K}, odd_factors={odd_factors}, sig1={sig1:x}, sig2={sig2:x}")
                breakpoint()
                continue
        
            # We want an additional shift to get the lsb into guard
            # So, lsb --> mantissa + 1
            trailing_zeros = len(bin(target_subnorm)) - len(bin(target_subnorm).rstrip('0'))
            lsb_location = bin(sig_quotient)[2:].rfind('1') + trailing_zeros
            required_shift = (common.MANTISSA_BITS[fmt] + 1) - lsb_location 

            # We want exp1 - exp2 + exponent_bias = -required_shift
            # so, exp2 = exp1 + exponent_bias + required_shift
            # -1 because softfloat
            exp2 = exp1 + common.EXPONENT_BIASES[fmt] + required_shift - 1

            if sig1 < sig2:
                exp2 -= 1

            in1 = generate_float(sign1, exp1, sig1 & ((1 << common.MANTISSA_BITS[fmt]) - 1), fmt)
            in2 = generate_float(sign2, exp2, sig2 & ((1 << common.MANTISSA_BITS[fmt]) - 1), fmt)

            tv = generate_test_vector(common.OP_DIV, in1, in2, 0, fmt, fmt, mode)
            result = run_test_vector(tv)

            info = extract_rounding_info(result)

            if info != target:
                print(info, target)
                print(f"Failure to generate exact division result, please investigate: K={K}, odd_factors={odd_factors}, sig1={sig1:x}, sig2={sig2:x}")
            else:
                store_cover_vector(result, test_f, cover_f)


def test_interm(fmt: str):
    cvt_ops = {
        common.OP_CFF: [common.FMT_HALF, common.FMT_SINGLE, common.FMT_DOUBLE, common.FMT_QUAD],
        common.OP_CIF: [common.FMT_INT, common.FMT_UINT, common.FMT_LONG, common.FMT_ULONG],
    }
    arith_ops_1src = [
        common.OP_SQRT,
    ]
    arith_ops_2src = [
        common.OP_ADD,
        common.OP_DIV,
        common.OP_MUL,
        common.OP_SUB,
        common.OP_REM,
    ]

    for op in cvt_ops:
        for target_fmt in cvt_ops[op]:
            if target_fmt == fmt:
                continue

            if fmt == common.FMT_BF16 and target_fmt in [common.FMT_LONG, common.FMT_ULONG]:
                continue

            for _ in range(10000):
                cvt_from = 0
                if target_fmt.startswith("0"):
                    cvt_from = generate_random_float(10, target_fmt)
                else:
                    cvt_from = random.randint(0, (1 << common.INT_SIZES[target_fmt]) - 1)
                    if random.random() < 0.5:
                        cvt_from = random.randint(0, 1000)

                tv = generate_test_vector(op, cvt_from, 0, 0, target_fmt, fmt)
                result = run_test_vector(tv)

                starting_sig = 0
                if target_fmt.startswith("0"):
                    starting_sig = bin(get_significand_from_float(cvt_from, target_fmt))[3:].rstrip('0')
                else:
                    starting_sig = cvt_from
                    if target_fmt in [common.FMT_INT, common.FMT_LONG] and starting_sig & (1 << (common.INT_SIZES[target_fmt] - 1)):
                        starting_sig = -(starting_sig - 2 ** (common.INT_SIZES[target_fmt]))
                    starting_sig = bin(starting_sig)[3:] # no leading one

                first_digit = result.split('_')[-1][0]
                ending_sig = bin(int(result.split('_')[-1], 16))[2:].zfill(192)

                if (starting_sig[:-2] in ending_sig) or (fmt == common.FMT_BF16): # and int(first_digit, 16) & 0x4 or (cvt_from == 0 and ending_sig == 0):
                    # print("success")
                    pass
                else:
                    breakpoint()

    for op in arith_ops_2src + arith_ops_1src:
        a = generate_random_float(common.MANTISSA_BITS[fmt] + 1, fmt)
        b = generate_random_float(0, fmt)

        tv = generate_test_vector(op, a, b, 0, fmt, fmt)
        result = run_test_vector(tv)

        mantissa = int(result.split('_')[-1], 16)
        sigA = get_significand_from_float(a, fmt)
        sigB = get_significand_from_float(b, fmt)

        should_be = bin(sigA)[2:] + bin(sigB)[2:]

        if not bin(mantissa)[2:].startswith(should_be):
            # breakpoint()
            pass


def main():
    # wtf = "00000010_00_00000000000000000000000040d76447_00000000000000000000000041d8af1d_00000000000000000000000000000000_01_00000000000000000000000042074417_01_01_0_00000084_362bc7400000000100000000000000000000000000000000"
    # extract_rounding_info(run_test_vector(wtf))
    # wtf2 = "00000010_00_0000000000000000000000000000431d_0000000000000000000000000000438e_00000000000000000000000000000000_00_00000000000000000000000000004756_00_01_0_00000011_1bd9f4c00000000000000000000000000000000000000000"
    # extract_rounding_info(run_test_vector(wtf2))
    # wtf3 = "00000020_00_4003b7b0646ec982599ca4e0eb60c4d1_4003aa281bc2af3f8a20282337bb1a19_00000000000000000000000000000000_03_3ffeb10915834859ef8f97b674b55700_03_00_0_00003ffe_0000b10915834859ef8f97b674b557000000000000000000"
    # extract_rounding_info(run_test_vector(wtf3))
    # wtf4 = "00000030_00_00000000000000004017e69d7cd24c38_00000000000000004021acd9f7533664_00000000000000000000000000000000_02_0000000000000000404a673c0ac99f5b_02_01_0_00000404_34ce7815933eb53d213b8c32bc0000000000000000000000"
    # extract_rounding_info(run_test_vector(wtf4))
    # wtf5 = "00000051_00_00000000000000004019d73273949a76_000000000000000040091b66a89ee668_000000000000000040426c256aa9fb52_02_0000000000000000404c8f455c263f49_02_01_0_00000404_288c7fc5f10fdd9986f3bf8c3f0000000000000000000000"
    # extract_rounding_info(run_test_vector(wtf5))

    # wtf = "00000051_03_00000000000000000000000000003f08_00000000000000000000000000003b8c_0000000000000000000000000000bea2_00_00000000000000000000000000000300_00_00_0_00000000_800000000000000000000000000000000000000000000000"
    # wtf = "00000010_00_c00b45ab35852b25b8dc32ac3eacc3f9_c009317f228339a481d4cc0c7c9ca0f0_00000000000000000000000000000000_03_c00b920afe25f98ed95165af5dd3ec35_03_00_1_0000400b_0006482bf897e63b654596bd774fb0d40000000000000000"
    # wtf = "00000010_00_3ffb11abfeb602871aa4cf914beb8c47_3ff962f214f9621999c4005e9e55bed0_00000000000000000000000000000000_03_3ffb6a6883f45b0d8115cfa8f380fbfb_03_00_0_00003ffb_b53441fa2d86c08ae7d479c07dfd80000000000000000000"
    # result = run_test_vector(wtf)
    # breakpoint()



    test_f = open("./tests/testvectors/B3_tv.txt", "w")
    cover_f = open("./tests/covervectors/B3_cv.txt", "w")

    # FMTS = [common.FMT_BF16, common.FMT_HALF, common.FMT_SINGLE, common.FMT_DOUBLE]
    # for fmt in FMTS:
    #     write_fma_tests(test_f, cover_f, fmt)
    #     test_interm(fmt)
    # return

    # These are going to be for sticky = 0
    for fmt in FMTS:
        write_add_sub_tests(test_f, cover_f, fmt)
        write_mul_tests(test_f, cover_f, fmt)
        write_div_tests(test_f, cover_f, fmt)
        write_sqrt_tests(test_f, cover_f, fmt)
        write_fma_tests(test_f, cover_f, fmt)
        test_interm(fmt)

    targets = [
        {
            'Sign': (x & 1),
            'LSB': (x & 2) >> 1,
            'Guard': (x & 4) >> 2,
            'Sticky': (x & 8) >> 3, # Sticky is one for all of these
        }
        for x in range(8, 16)
    ]

    targets = [x for x in targets if x['Sign'] == 0]

    misses = 0
    emmitted = 0
    total = 0

    for op in [*SRC1_OPS, *SRC2_OPS, *SRC3_OPS]:
        for fmt in FMTS:
            for mode in common.ROUNDING_MODES:
                cover_goals = targets[:]
                if op == common.OP_SQRT or op == common.OP_REM:
                    cover_goals = [x for x in cover_goals if x['Sign'] == 0]

                for _ in range(200):
                    in1 = generate_random_float(random.randint(0, 5), fmt)
                    in2 = generate_random_float(random.randint(0, 5), fmt) if op in SRC2_OPS or op in SRC3_OPS else 0
                    in3 = generate_random_float(random.randint(0, 5), fmt) if op in SRC3_OPS else 0

                    tv = generate_test_vector(op, in1, in2, in3, fmt, fmt, mode)
                    cv = run_test_vector(tv)

                    # if op == OP_REM:
                    #     breakpoint()

                    rounding_results = extract_rounding_info(cv)

                    if rounding_results in cover_goals:
                        cover_goals.remove(rounding_results)
                        print(cv[:common.TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS], file=test_f)
                        print(cv, file=cover_f)
                        emmitted += 1
                    
                    if len(cover_goals) == 0:
                        break
                else:
                    print("Miss: ", op, fmt, len(cover_goals), cover_goals)
                    misses += len(cover_goals)
                total += len(targets)

    print(f"Hit rate: {emmitted/total}, {emmitted}, {total}")

if __name__ == "__main__":
    main()

"""
in1 = generate_random_float32(1)
in2 = generate_random_float32(3)

tv = generate_test_vector(OP_ADD, in1, in2, FMT_SINGLE, FMT_SINGLE)
cv = coverfloat.reference(tv)

info = extract_rounding_info(cv)
"""