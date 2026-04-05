#include "coverfloat.hpp"
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace mp = boost::multiprecision;

void softFloat_clearFlags(uint_fast8_t clearMask) {
    softfloat_exceptionFlags &= ~clearMask;
}

uint_fast8_t softFloat_getFlags() {
    return softfloat_exceptionFlags;
}

void softFloat_setRoundingMode(uint_fast8_t rm) {
    softfloat_roundingMode = rm;
}

void softfloat_getIntermResults(intermResult_t *result) {

    result->sign = softfloat_intermediateResult.sign;
    result->exp = softfloat_intermediateResult.exp;
    result->sig64 = softfloat_intermediateResult.sig64;
    result->sig0 = softfloat_intermediateResult.sig0;
    result->sigExtra64 = softfloat_intermediateResult.sigExtra64;
    result->sigExtra0 = softfloat_intermediateResult.sigExtra0;

    result->fmaPreAddition[indexWord(4, 3)] = softfloat_intermediateResult.fmaPreAddition[indexWord(4, 3)];
    result->fmaPreAddition[indexWord(4, 2)] = softfloat_intermediateResult.fmaPreAddition[indexWord(4, 2)];
    result->fmaPreAddition[indexWord(4, 1)] = softfloat_intermediateResult.fmaPreAddition[indexWord(4, 1)];
    result->fmaPreAddition[indexWord(4, 0)] = softfloat_intermediateResult.fmaPreAddition[indexWord(4, 0)];
}

void softfloat_getIntermResults(MPIntermResult &result) {
    result.sign = softfloat_intermediateResult.sign;
    result.exp = softfloat_intermediateResult.exp;

    result.sig = softfloat_intermediateResult.sig64;
    result.sig <<= 64;
    result.sig |= softfloat_intermediateResult.sig0;
    result.sig <<= 64;
    result.sig |= softfloat_intermediateResult.sigExtra64;
    result.sig <<= 64;
    result.sig |= softfloat_intermediateResult.sigExtra0;

    result.fma_pre_addition = softfloat_intermediateResult.fmaPreAddition[indexWord(4, 3)];
    result.fma_pre_addition <<= 64;
    result.fma_pre_addition |= softfloat_intermediateResult.fmaPreAddition[indexWord(4, 2)];
    result.fma_pre_addition <<= 64;
    result.fma_pre_addition |= softfloat_intermediateResult.fmaPreAddition[indexWord(4, 1)];
    result.fma_pre_addition <<= 64;
    result.fma_pre_addition |= softfloat_intermediateResult.fmaPreAddition[indexWord(4, 0)];
}

void softfloat_clearIntermResults() {

    softfloat_intermediateResult.sign = 0;
    softfloat_intermediateResult.exp = 0;
    softfloat_intermediateResult.sig64 = 0;
    softfloat_intermediateResult.sig0 = 0;
    softfloat_intermediateResult.sigExtra64 = 0;
    softfloat_intermediateResult.sigExtra0 = 0;
    softfloat_intermediateResult.fmaPreAddition[indexWord(4, 0)] = 0;
    softfloat_intermediateResult.fmaPreAddition[indexWord(4, 1)] = 0;
    softfloat_intermediateResult.fmaPreAddition[indexWord(4, 2)] = 0;
    softfloat_intermediateResult.fmaPreAddition[indexWord(4, 3)] = 0;
}

/*
AI CODE NEEDS MODIFICATION
*/

/*
uint128_t parse_hex_128(const char *hex) {
    uint128_t value = {0, 0};

    while (*hex) {
        char c = *hex++;
        uint8_t digit;

        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
        else continue; // skip non-hex chars

        // Shift value left by 4 bits (multiply by 16)
        uint64_t new_upper = (value.upper << 4) | (value.lower >> 60);
        uint64_t new_lower = (value.lower << 4) | digit;

        value.upper = new_upper;
        value.lower = new_lower;
    }

    return value;
}
*/

uint128_t parse_hex_128(const char *hex) {
    uint128_t value = {(uint64_t)0ULL, (uint64_t)0ULL};
    int count = 0;

    while (*hex && count < 32) {
        char c = *hex++;
        uint8_t digit;

        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit = 10 + (c - 'a');
        } else if (c >= 'A' && c <= 'F') {
            digit = 10 + (c - 'A');
        } else {
            continue;
        }

        uint64_t upper = (uint64_t)value.upper;
        uint64_t lower = (uint64_t)value.lower;

        value.upper = (upper << 4) | (lower >> 60);
        value.lower = (lower << 4) | (uint64_t)digit;

        // printf("VALUE AT STEP %d: %016x%016x\n", count, value.upper, value.lower);

        count++;
    }

    return value;
}

int reference_model(
    const uint32_t op,
    const uint8_t rm,
    const mp::uint128_t &a,
    const mp::uint128_t &b,
    const mp::uint128_t &c,
    const uint8_t operandFmt,
    const uint8_t resultFmt,

    mp::uint128_t &result,
    uint8_t *flags,
    MPIntermResult &intermResult
) {

    // clear flags so we get only triggered flags
    softFloat_clearFlags(0xFF);

    // clear intermediate result to avoid reporting intermediate results for results that were not rounded
    softfloat_clearIntermResults();

    // set rounding mode
    softFloat_setRoundingMode(rm);

    // nested switch statements to call softfloat functions

    switch (op) {
    case OP_ADD: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf = f32_add(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf = f64_add(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf = f128_add(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = f16_add(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = bf16_add(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }
        break;
    }

    case OP_SUB: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf = f32_sub(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf = f64_sub(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf = f128_sub(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = f16_sub(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = bf16_sub(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_MUL: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf = f32_mul(af, bf);
            FLOAT32_TO_MP(result, resultf);

            // printf("performing single precision mul!!\n");
            // printf("int operands are: %x and %x\n", *a, *b);
            // printf("float operands are: %x and %x\n", af.v, bf.v);
            // printf("float result is %x\n", resultf.v);
            // printf("int result is %032x%032x\n", result->upper, result->lower);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf = f64_mul(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf = f128_mul(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = f16_mul(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = bf16_mul(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_DIV: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf = f32_div(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf = f64_div(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf = f128_div(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = f16_div(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = bf16_div(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_REM: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf = f32_rem(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf = f64_rem(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf = f128_rem(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = f16_rem(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        // TODO: not currently implemented as a function through softfloat
        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            // resultf = bf16_rem(af, bf);
            float32_t f32A = {(uint32_t)af.v << 16};
            float32_t f32B = {(uint32_t)bf.v << 16};
            resultf = f32_to_bf16(f32_div(f32A, f32B));

            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FEQ: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf.v = f32_eq(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf.v = f64_eq(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf.v[1] = 0;
            resultf.v[0] = f128_eq(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = f16_eq(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = bf16_eq(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FLT: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf.v = f32_lt(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf.v = f64_lt(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf.v[1] = 0;
            resultf.v[0] = f128_lt(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = f16_lt(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = bf16_lt(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FLE: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf.v = f32_le(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf.v = f64_le(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf.v[1] = 0;
            resultf.v[0] = f128_le(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = f16_le(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = bf16_le(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_MIN: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf = f32_min(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf = f64_min(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        // TODO: Missing softfloat function
        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf = f128_min(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = f16_min(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = bf16_min(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_MAX: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf = f32_max(af, bf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf = f64_max(af, bf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        // NOTE: Missing softfloat function, added custom
        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf = f128_max(af, bf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = f16_max(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf = bf16_max(af, bf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FSGNJ: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf.v = (af.v & 0x7FFFFFFF) | (bf.v & 0x80000000);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf.v = (af.v & 0x7FFFFFFFFFFFFFFF) | (bf.v & 0x8000000000000000);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf.v[1] = (af.v[1] & 0x7FFFFFFFFFFFFFFF) | (bf.v[1] & 0x8000000000000000);
            resultf.v[0] = af.v[0];
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = (af.v & 0x7FFF) | (bf.v & 0x8000);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = (af.v & 0x7FFF) | (bf.v & 0x8000);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FSGNJN: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf.v = (af.v & 0x7FFFFFFF) | (~(bf.v) & 0x80000000);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf.v = (af.v & 0x7FFFFFFFFFFFFFFF) | (~(bf.v) & 0x8000000000000000);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf.v[1] = (af.v[1] & 0x7FFFFFFFFFFFFFFF) | (~(bf.v[1]) & 0x8000000000000000);
            resultf.v[0] = af.v[0];
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = (af.v & 0x7FFF) | (~(bf.v) & 0x8000);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = (af.v & 0x7FFF) | (~(bf.v) & 0x8000);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FSGNJX: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            resultf.v = (af.v & 0x7FFFFFFF) | ((af.v ^ bf.v) & 0x80000000);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            resultf.v = (af.v & 0x7FFFFFFFFFFFFFFF) | ((af.v ^ bf.v) & 0x8000000000000000);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            resultf.v[1] = (af.v[1] & 0x7FFFFFFFFFFFFFFF) | ((af.v[1] ^ bf.v[1]) & 0x8000000000000000);
            resultf.v[0] = af.v[0];
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = (af.v & 0x7FFF) | ((af.v ^ bf.v) & 0x8000);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            resultf.v = (af.v & 0x7FFF) | ((af.v ^ bf.v) & 0x8000);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_CIF: {
        switch (resultFmt) {
        case FMT_HALF: {
            float16_t out;
            switch (operandFmt) {
            case FMT_INT: {
                uint32_t serialized_input = static_cast<uint32_t>(a);
                int32_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i32_to_f16(input);
                break;
            }
            case FMT_UINT: {
                uint32_t input = static_cast<uint32_t>(a);
                out = ui32_to_f16(input);
                break;
            }
            case FMT_LONG: {
                uint64_t serialized_input = static_cast<uint64_t>(a);
                int64_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i64_to_f16(input);
                break;
            }
            case FMT_ULONG: {
                uint64_t input = static_cast<uint64_t>(a);
                out = ui64_to_f16(input);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: int to float conversion with non-int operand format: %x\n", operandFmt);
                return EXIT_FAILURE;
            }
            }

            FLOAT16_TO_MP(result, out);
            break;
        }
        case FMT_BF16: {
            float16_t out;
            switch (operandFmt) {
            case FMT_INT: {
                uint32_t serialized_input = static_cast<uint32_t>(a);
                int32_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i32_to_bf16(input);
                break;
            }
            case FMT_UINT: {
                uint32_t input = static_cast<uint32_t>(a);
                out = ui32_to_bf16(input);
                break;
            }
            case FMT_LONG: {
                uint64_t serialized_input = static_cast<uint64_t>(a);
                int64_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int64_t is UB
                memcpy(&input, &serialized_input, sizeof(input));

                softFloat_setRoundingMode(softfloat_round_odd);
                float64_t out_64 = i64_to_f64(input);
                softFloat_setRoundingMode(rm);
                out = f64_to_bf16(out_64);

                break;
            }
            case FMT_ULONG: {
                uint64_t input = static_cast<uint64_t>(a);

                softFloat_setRoundingMode(softfloat_round_odd);
                float64_t out_64 = ui64_to_f64(input);
                softFloat_setRoundingMode(rm);
                out = f64_to_bf16(out_64);

                break;
            }
            default: {
                fprintf(stderr, "ERROR: int to float conversion with unsupported operand format: %x\n", operandFmt);
                return EXIT_FAILURE;
            }
            }

            FLOAT16_TO_MP(result, out);
            break;
        }
        case FMT_SINGLE: {
            float32_t out;
            switch (operandFmt) {
            case FMT_INT: {
                uint32_t serialized_input = static_cast<uint32_t>(a);
                int32_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i32_to_f32(input);
                break;
            }
            case FMT_UINT: {
                uint32_t input = static_cast<uint32_t>(a);
                out = ui32_to_f32(input);
                break;
            }
            case FMT_LONG: {
                uint64_t serialized_input = static_cast<uint64_t>(a);
                int64_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i64_to_f32(input);
                break;
            }
            case FMT_ULONG: {
                uint64_t input = static_cast<uint64_t>(a);
                out = ui64_to_f32(input);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: int to float conversion with non-int operand format: %x\n", operandFmt);
                return EXIT_FAILURE;
            }
            }

            FLOAT32_TO_MP(result, out);
            break;
        }
        case FMT_DOUBLE: {
            float64_t out;
            switch (operandFmt) {
            case FMT_INT: {
                uint32_t serialized_input = static_cast<uint32_t>(a);
                int32_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i32_to_f64(input);
                break;
            }
            case FMT_UINT: {
                uint32_t input = static_cast<uint32_t>(a);
                out = ui32_to_f64(input);
                break;
            }
            case FMT_LONG: {
                uint64_t serialized_input = static_cast<uint64_t>(a);
                int64_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i64_to_f64(input);
                break;
            }
            case FMT_ULONG: {
                uint64_t input = static_cast<uint64_t>(a);
                out = ui64_to_f64(input);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: int to float conversion with non-int operand format: %x\n", operandFmt);
                return EXIT_FAILURE;
            }
            }

            FLOAT64_TO_MP(result, out);
            break;
        }
        case FMT_QUAD: {
            float128_t out;
            switch (operandFmt) {
            case FMT_INT: {
                uint32_t serialized_input = static_cast<uint32_t>(a);
                int32_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i32_to_f128(input);
                break;
            }
            case FMT_UINT: {
                uint32_t input = static_cast<uint32_t>(a);
                out = ui32_to_f128(input);
                break;
            }
            case FMT_LONG: {
                uint64_t serialized_input = static_cast<uint64_t>(a);
                int64_t input;
                // We need to be careful not to throw out the signed part of it
                // a direct conversion to int32_t is UB
                memcpy(&input, &serialized_input, sizeof(input));
                out = i64_to_f128(input);
                break;
            }
            case FMT_ULONG: {
                uint64_t input = static_cast<uint64_t>(a);
                out = ui64_to_f128(input);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: int to float conversion with non-int operand format: %x\n", operandFmt);
                return EXIT_FAILURE;
            }
            }

            FLOAT128_TO_MP(result, out);
            break;
        }
        default: {
            fprintf(stderr, "ERROR: int to float conversion called with unsupported result format: %x\n", resultFmt);
            return EXIT_FAILURE;
        }
        }

        break;
    }

    case OP_CFI: {
        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af;
            // TODO: maybe sign extend...
            MP_TO_FLOAT32(af, a);
            switch (resultFmt) {
            case FMT_INT: {
                result = signed_to_unsigned(f32_to_i32(af, rm, true));
                break;
            }
            case FMT_UINT: {
                result = f32_to_ui32(af, rm, true);
                break;
            }
            case FMT_LONG: {
                result = signed_to_unsigned(f32_to_i64(af, rm, true));
                break;
            }
            case FMT_ULONG: {
                result = f32_to_ui64(af, rm, true);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to int conversion with non-int result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af;
            MP_TO_FLOAT64(af, a);
            switch (resultFmt) {
            case FMT_INT: {
                result = signed_to_unsigned(f64_to_i32(af, rm, true));
                break;
            }
            case FMT_UINT: {
                result = f64_to_ui32(af, rm, true);
                break;
            }
            case FMT_LONG: {
                result = signed_to_unsigned(f64_to_i64(af, rm, true));
                break;
            }
            case FMT_ULONG: {
                result = f64_to_ui64(af, rm, true);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to int conversion with non-int result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af;
            MP_TO_FLOAT128(af, a);
            switch (resultFmt) {
            case FMT_INT: {
                result = signed_to_unsigned(f128_to_i32(af, rm, true));
                break;
            }
            case FMT_UINT: {
                result = f128_to_ui32(af, rm, true);
                break;
            }
            case FMT_LONG: {
                result = signed_to_unsigned(f128_to_i64(af, rm, true));
                break;
            }
            case FMT_ULONG: {
                result = f128_to_ui64(af, rm, true);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to int conversion with non-int result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af;
            MP_TO_FLOAT16(af, a);
            switch (resultFmt) {
            case FMT_INT: {
                result = signed_to_unsigned(f16_to_i32(af, rm, true));
                break;
            }
            case FMT_UINT: {
                result = f16_to_ui32(af, rm, true);
                break;
            }
            case FMT_LONG: {
                result = signed_to_unsigned(f16_to_i64(af, rm, true));
                break;
            }
            case FMT_ULONG: {
                result = f16_to_ui64(af, rm, true);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to int conversion with non-int result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af;
            MP_TO_FLOAT16(af, a);
            switch (resultFmt) {
            case FMT_INT: {
                result = signed_to_unsigned(bf16_to_i32(af, rm, true));
                break;
            }
            case FMT_UINT: {
                result = bf16_to_ui32(af, rm, true);
                break;
            }
            case FMT_LONG: {
                result = f32_to_i64(bf16_to_f32(af), rm, true);
                break;
            }
            case FMT_ULONG: {
                result = f32_to_ui64(bf16_to_f32(af), rm, true);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to int conversion with non-int result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT16_TO_MP(result, resultf);
            break;
        }
        default: {
            fprintf(stderr, "ERROR: float to int conversion with non-float source format\n");
        }
        }

        break;
    }

    /* TODO: float to float instructions */
    case OP_CFF: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af;
            MP_TO_FLOAT32(af, a);
            switch (resultFmt) {
            case FMT_HALF: {
                float16_t resultf = f32_to_f16(af);
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_BF16: {
                bfloat16_t resultf = f32_to_bf16(af);
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_SINGLE: {
                fprintf(stderr, "ERROR: float to float conversion with the same operand and result format (single)\n");
                return EXIT_FAILURE;
            }
            case FMT_DOUBLE: {
                float64_t resultf = f32_to_f64(af);
                FLOAT64_TO_MP(result, resultf);
                break;
            }
            case FMT_QUAD: {
                float128_t resultf = f32_to_f128(af);
                FLOAT128_TO_MP(result, resultf);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to float conversion with non-float result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af;
            MP_TO_FLOAT64(af, a);
            switch (resultFmt) {
            case FMT_HALF: {
                float16_t resultf = f64_to_f16(af);
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_BF16: {
                bfloat16_t resultf = f64_to_bf16(af);
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_SINGLE: {
                float32_t resultf = f64_to_f32(af);
                FLOAT32_TO_MP(result, resultf);
                break;
            }
            case FMT_DOUBLE: {
                fprintf(stderr, "ERROR: float to float conversion with the same operand and result format (double)\n");
                return EXIT_FAILURE;
            }
            case FMT_QUAD: {
                float128_t resultf = f64_to_f128(af);
                FLOAT128_TO_MP(result, resultf);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to float conversion with non-float result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af;
            MP_TO_FLOAT128(af, a);
            switch (resultFmt) {
            case FMT_HALF: {
                float16_t resultf = f128_to_f16(af);
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_BF16: {
                softFloat_setRoundingMode(softfloat_round_odd);
                float32_t af_32 = f128_to_f32(af);
                softFloat_setRoundingMode(rm);

                bfloat16_t resultf = f32_to_bf16(af_32);
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_SINGLE: {
                float32_t resultf = f128_to_f32(af);
                FLOAT32_TO_MP(result, resultf);
                break;
            }
            case FMT_DOUBLE: {
                float64_t resultf = f128_to_f64(af);
                FLOAT64_TO_MP(result, resultf);
                break;
            }
            case FMT_QUAD: {
                fprintf(stderr, "ERROR: float to float conversion with the same operand and result format (quad)\n");
                return EXIT_FAILURE;
            }
            default: {
                fprintf(stderr, "ERROR: float to float conversion with non-float result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af;
            MP_TO_FLOAT16(af, a);
            switch (resultFmt) {
            case FMT_HALF: {
                fprintf(stderr, "ERROR: float to float conversion with the same operand and result format (half)\n");
                return EXIT_FAILURE;
            }
            case FMT_BF16: {
                bfloat16_t resultf = f32_to_bf16(f16_to_f32(af));
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_SINGLE: {
                float32_t resultf = f16_to_f32(af);
                FLOAT32_TO_MP(result, resultf);
                break;
            }
            case FMT_DOUBLE: {
                float64_t resultf = f16_to_f64(af);
                FLOAT64_TO_MP(result, resultf);
                break;
            }
            case FMT_QUAD: {
                float128_t resultf = f16_to_f128(af);
                FLOAT128_TO_MP(result, resultf);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to float conversion with non-float result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af;
            MP_TO_FLOAT16(af, a);
            switch (resultFmt) {
            case FMT_HALF: {
                float16_t resultf = f32_to_f16(bf16_to_f32(af));
                FLOAT16_TO_MP(result, resultf);
                break;
            }
            case FMT_BF16: {
                fprintf(stderr, "ERROR: float to float conversion with the same operand and result format (half)\n");
                return EXIT_FAILURE;
            }
            case FMT_SINGLE: {
                float32_t resultf = bf16_to_f32(af);
                FLOAT32_TO_MP(result, resultf);
                break;
            }
            case FMT_DOUBLE: {
                float64_t resultf = f32_to_f64(bf16_to_f32(af));
                FLOAT64_TO_MP(result, resultf);
                break;
            }
            case FMT_QUAD: {
                float128_t resultf = f32_to_f128(bf16_to_f32(af));
                FLOAT128_TO_MP(result, resultf);
                break;
            }
            default: {
                fprintf(stderr, "ERROR: float to float conversion with non-float result format\n");
                return EXIT_FAILURE;
            }
            }
            // FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_SQRT: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, resultf;
            MP_TO_FLOAT32(af, a);
            resultf = f32_sqrt(af);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, resultf;
            MP_TO_FLOAT64(af, a);
            resultf = f64_sqrt(af);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, resultf;
            MP_TO_FLOAT128(af, a);
            resultf = f128_sqrt(af);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, resultf;
            MP_TO_FLOAT16(af, a);
            resultf = f16_sqrt(af);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, resultf;
            MP_TO_FLOAT16(af, a);
            resultf = bf16_sqrt(af);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_CLASS: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, resultf;
            MP_TO_FLOAT32(af, a);
            resultf.v = f32_classify(af);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, resultf;
            MP_TO_FLOAT64(af, a);
            resultf.v = f64_classify(af);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, resultf;
            MP_TO_FLOAT128(af, a);
            resultf.v[1] = 0;
            resultf.v[0] = f128_classify(af);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, resultf;
            MP_TO_FLOAT16(af, a);
            resultf.v = f16_classify(af);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, resultf;
            MP_TO_FLOAT16(af, a);
            resultf.v = bf16_classify(af);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FMADD: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, cf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            MP_TO_FLOAT32(cf, c);
            resultf = f32_mulAdd(af, bf, cf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, cf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            MP_TO_FLOAT64(cf, c);
            resultf = f64_mulAdd(af, bf, cf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, cf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            MP_TO_FLOAT128(cf, c);
            resultf = f128_mulAdd(af, bf, cf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            resultf = f16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            resultf = bf16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FMSUB: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, cf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            MP_TO_FLOAT32(cf, c);
            cf.v ^= 0x80000000; // flip sign
            resultf = f32_mulAdd(af, bf, cf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, cf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            MP_TO_FLOAT64(cf, c);
            cf.v ^= 0x8000000000000000; // flip sign
            resultf = f64_mulAdd(af, bf, cf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, cf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            MP_TO_FLOAT128(cf, c);
            cf.v[1] ^= 0x8000000000000000; // flip sign
            resultf = f128_mulAdd(af, bf, cf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            cf.v ^= 0x8000; // flip sign
            resultf = f16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            cf.v ^= 0x8000; // flip sign
            resultf = bf16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FNMADD: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, cf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            MP_TO_FLOAT32(cf, c);
            af.v ^= 0x80000000; // flip sign
            cf.v ^= 0x80000000; // flip sign
            resultf = f32_mulAdd(af, bf, cf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, cf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            MP_TO_FLOAT64(cf, c);
            af.v ^= 0x8000000000000000; // flip sign
            cf.v ^= 0x8000000000000000; // flip sign
            resultf = f64_mulAdd(af, bf, cf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, cf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            MP_TO_FLOAT128(cf, c);
            af.v[1] ^= 0x8000000000000000; // flip sign
            cf.v[1] ^= 0x8000000000000000; // flip sign
            resultf = f128_mulAdd(af, bf, cf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            af.v ^= 0x8000; // flip sign
            cf.v ^= 0x8000; // flip sign
            resultf = f16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            af.v ^= 0x8000; // flip sign
            cf.v ^= 0x8000; // flip sign
            resultf = bf16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    case OP_FNMSUB: {

        switch (operandFmt) {
        case FMT_SINGLE: {
            float32_t af, bf, cf, resultf;
            MP_TO_FLOAT32(af, a);
            MP_TO_FLOAT32(bf, b);
            MP_TO_FLOAT32(cf, c);
            af.v ^= 0x80000000; // flip sign
            resultf = f32_mulAdd(af, bf, cf);
            FLOAT32_TO_MP(result, resultf);
            break;
        }

        case FMT_DOUBLE: {
            float64_t af, bf, cf, resultf;
            MP_TO_FLOAT64(af, a);
            MP_TO_FLOAT64(bf, b);
            MP_TO_FLOAT64(cf, c);
            af.v ^= 0x8000000000000000; // flip sign
            resultf = f64_mulAdd(af, bf, cf);
            FLOAT64_TO_MP(result, resultf);
            break;
        }

        case FMT_QUAD: {
            float128_t af, bf, cf, resultf;
            MP_TO_FLOAT128(af, a);
            MP_TO_FLOAT128(bf, b);
            MP_TO_FLOAT128(cf, c);
            af.v[1] ^= 0x8000000000000000; // flip sign
            resultf = f128_mulAdd(af, bf, cf);
            FLOAT128_TO_MP(result, resultf);
            break;
        }

        case FMT_HALF: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            af.v ^= 0x8000; // flip sign
            resultf = f16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }

        case FMT_BF16: {
            float16_t af, bf, cf, resultf;
            MP_TO_FLOAT16(af, a);
            MP_TO_FLOAT16(bf, b);
            MP_TO_FLOAT16(cf, c);
            af.v ^= 0x8000; // flip sign
            resultf = bf16_mulAdd(af, bf, cf);
            FLOAT16_TO_MP(result, resultf);
            break;
        }
        }

        break;
    }

    default: {
        fprintf(stderr, "Unsupported Operation Called, OP: %x\n", op);
        return EXIT_FAILURE;
    }
    }

    *flags = softFloat_getFlags();
    softfloat_getIntermResults(intermResult);

    if (op == OP_CFI && operandFmt == FMT_HALF && (resultFmt == FMT_LONG || resultFmt == FMT_ULONG)) {
        intermResult.sig >>= 32;
    }

    if (intermResult.exp == 0 && intermResult.sig == 0) {
        // Then we need to extract an intermediate result from the result
        switch (resultFmt) {
        case FMT_BF16: {
            uint64_t sig = static_cast<uint64_t>(fracBF16UI(result));
            uint32_t exp = expBF16UI(result);

            // No leading one if subnorm or zero
            if (exp != 0) {
                sig |= BF16_IMPLICIT_ONE;
            }
            intermResult.sig = mp::uint256_t(sig) << (INTERM_SIG_LENGTH - 2 - BF16_SIG_BITS);

            intermResult.exp = exp;

            intermResult.sign = signBF16UI(result);
            break;
        }
        case FMT_HALF: {
            uint64_t sig = static_cast<uint64_t>(fracF16UI(result));
            uint32_t exp = expF16UI(result);

            // No leading one if it is a subnorm or a zero
            if (exp != 0) {
                sig |= (1 << 10);
            }

            intermResult.sig = mp::uint256_t(sig) << (INTERM_SIG_LENGTH - 2 - 10);

            intermResult.exp = exp;

            intermResult.sign = signF16UI(result);
            break;
        }
        case FMT_SINGLE: {
            uint64_t sig = static_cast<uint64_t>(fracF32UI(result));
            uint32_t exp = expF32UI(result);

            if (exp != 0) {
                sig |= (1 << 23);
                intermResult.sig = mp::uint256_t(sig) << (INTERM_SIG_LENGTH - 2 - 23);
            }

            intermResult.exp = exp;

            intermResult.sign = signF32UI(result);
            break;
        }
        case FMT_DOUBLE: {
            uint64_t sig = static_cast<uint64_t>(fracF64UI(result));
            uint32_t exp = expF64UI(result);

            if (exp != 0) {
                sig |= (1UL << 52);
            }

            intermResult.sig = mp::uint256_t(sig) << (INTERM_SIG_LENGTH - 2 - 52);

            intermResult.exp = exp;

            intermResult.sign = signF64UI(result);
            break;
        }
        case FMT_QUAD: {
            mp::uint256_t sig = fracF128UI64(result >> 64);
            sig <<= 64;
            sig |= static_cast<uint64_t>(result);
            uint32_t exp = expF128UI64(result >> 64);

            // Exp = 0 is a subnorm or a zero
            if (exp != 0) {
                sig |= ((mp::uint128_t)1) << 112;
            }
            // intermResult.sig64 = static_cast<uint64_t>(sig >> 64);
            // intermResult.sig0 = static_cast<uint64_t>(sig);
            intermResult.sig = sig << (INTERM_SIG_LENGTH - 128);

            intermResult.exp = exp;

            intermResult.sign = signF128UI64(result >> 64);
            break;
        }

        default:
            // Int format
            break;
        }
    }

    // If we took an fma operation, there is now an alignment shift that needs to take place
    if ((op & ~(0xf)) == OP_FMA) {
        if (operandFmt == FMT_SINGLE) {
            intermResult.fma_pre_addition >>= 14;
        } else if (operandFmt == FMT_DOUBLE) {
            intermResult.fma_pre_addition >>= 20;
        } else if (operandFmt == FMT_QUAD) {
            intermResult.fma_pre_addition >>= 23;
        } else if (operandFmt == FMT_HALF) {
            intermResult.fma_pre_addition >>= 8;
        } else if (operandFmt == FMT_BF16) {
            intermResult.fma_pre_addition >>= 38;
        }
    }

    // Post-process the intermediate results:
    // 1. Ensure that subnorms have everything in the right place
    // 2. Then shift off the leading ones

    // 1
    if (intermResult.exp <= 0) {
        // See s_roundPackToF32.c for why we add 1. Our exp is +1 theirs
        int32_t shift_dist = -intermResult.exp + 1;

        // Unfortunately, softfloat doesn't give us a 192-bit right shift jam
        // but look at softfloat_shiftRightJam128() for reference

        /*
        if (shift_dist < 64) {
            // Everything has a short shift here, most complex case, but look at reference from softfloat
            intermResult->sigExtra = (intermResult->sigExtra >> shift_dist) |
                                     (intermResult->sig0 << (64 - shift_dist)) |
                                     ((intermResult->sigExtra << (64 - shift_dist)) != 0); // This is the jam
            intermResult->sig0 = (intermResult->sig0 >> shift_dist) | (intermResult->sig64 << (64 - shift_dist));
            intermResult->sig64 = intermResult->sig64 >> shift_dist;
        } else if (shift_dist < 128) {
            // These two cases are the same as above but simpler
            intermResult->sigExtra =
                (intermResult->sig0 >> (shift_dist - 64)) |
                (((intermResult->sig0 << (128 - shift_dist)) | intermResult->sigExtra) != 0); // This is the jam
            intermResult->sig0 = intermResult->sig64 >> (shift_dist - 64);
            intermResult->sig64 = 0;
        } else if (shift_dist < 192) {
            intermResult->sigExtra = (intermResult->sig64 >> (shift_dist - 128)) |
                                     (((intermResult->sig64 << (192 - shift_dist)) | intermResult->sigExtra |
                                       intermResult->sig0) != 0); // This is the jam
            intermResult->sig0 = 0;
            intermResult->sig64 = 0;
        } else {
            // Only a jam here
            intermResult->sigExtra = (intermResult->sig0 | intermResult->sig64 | intermResult->sigExtra) != 0;
            intermResult->sig0 = 0;
            intermResult->sig64 = 0;
        }
        */

        // Lets shift right jam ourselves now!
        mp::uint256_t mask = (mp::uint256_t(1) << shift_dist) - 1;
        int should_jam = 0;
        if (intermResult.sig & mask) {
            should_jam = 1;
        }

        intermResult.sig >>= shift_dist;
        intermResult.sig |= should_jam;
        intermResult.exp = 0;
    }

    // 2
    uint8_t shift_amount = (resultFmt == FMT_QUAD) ? 16 : 2;
    // struct uint128 shifted_sig = softfloat_shortShiftLeft128(intermResult->sig64, intermResult->sig0, shift_amount);

    // intermResult->sig64 = shifted_sig.v64;
    // intermResult->sig0 =
    //     shifted_sig.v0 | (intermResult->sigExtra >> (-shift_amount & 63)); // Look at shortShiftLeft source
    // intermResult->sigExtra = intermResult->sigExtra << shift_amount;
    intermResult.sig <<= shift_amount;
    intermResult.sig &= ~mp::uint256_t(0);

    return EXIT_SUCCESS;
}

// TODO move to own file
float128_t f128_min(float128_t a, float128_t b) {
    bool less = f128_lt_quiet(a, b) || (f128_eq(a, b) && signF128UI64(a.v[1]));

    if (isNaNF128UI(a.v[1], a.v[0]) && isNaNF128UI(b.v[1], b.v[0])) {
        union ui128_f128 ui;
        ui.ui.v64 = defaultNaNF128UI64;
        ui.ui.v0 = defaultNaNF128UI0;
        return ui.f;
    } else {
        return (less || isNaNF128UI(b.v[0], b.v[1])) ? a : b;
    }
}

float128_t f128_max(float128_t a, float128_t b) {
    bool greater = f128_lt_quiet(b, a) || (f128_eq(b, a) && signF128UI64(b.v[1]));

    if (isNaNF128UI(a.v[1], a.v[0]) && isNaNF128UI(b.v[1], b.v[0])) {
        union ui128_f128 ui;
        ui.ui.v64 = defaultNaNF128UI64;
        ui.ui.v0 = defaultNaNF128UI0;
        return ui.f;
    } else {
        return (greater || isNaNF128UI(b.v[0], b.v[1])) ? a : b;
    }
}

std::string coverfloat_runtestvector(const std::string &input, bool suppress_error_check) {

#if 0
    char op_str[MAX_TOKEN_LEN + 1]; // plus one for space for null terminator
    char rm_str[MAX_TOKEN_LEN + 1];
    char a_str[MAX_TOKEN_LEN + 1];
    char b_str[MAX_TOKEN_LEN + 1];
    char c_str[MAX_TOKEN_LEN + 1];
    char opFmt_str[MAX_TOKEN_LEN + 1];
    char res_str[MAX_TOKEN_LEN + 1];
    char resFmt_str[MAX_TOKEN_LEN + 1];
    char flags_str[MAX_TOKEN_LEN + 1];

    if (sscanf(
            input,
            "%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_ \t\r\n]",
            op_str,
            rm_str,
            a_str,
            b_str,
            c_str,
            opFmt_str,
            res_str,
            resFmt_str,
            flags_str
        ) != 9) {
        snprintf(output, output_size, "Error: malformed testvector: %s\n", input);

        return EXIT_FAILURE;
    }

    // unpack test vector tokens into integers to pass to the reference model

    uint32_t op = parse_hex_128(op_str).lower;
    uint8_t rm = parse_hex_128(rm_str).lower;
    uint128_t a = parse_hex_128(a_str);
    uint128_t b = parse_hex_128(b_str);
    uint128_t c = parse_hex_128(c_str);
    uint8_t opFmt = parse_hex_128(opFmt_str).lower;
    uint8_t resFmt = parse_hex_128(resFmt_str).lower;
    uint128_t res = parse_hex_128(res_str);
    uint8_t flags = parse_hex_128(flags_str).lower;
#else
    std::stringstream ss(input);
    ss >> std::hex;

    uint32_t op;
    uint16_t rm16;
    mp::uint128_t a, b, c;
    uint16_t opFmt16;
    mp::uint128_t res;
    uint16_t resFmt16;
    uint16_t flags;

    ss >> op;
    ss.ignore(1);
    ss >> rm16;
    ss.ignore(1);
    ss >> a;
    ss.ignore(1);
    ss >> b;
    ss.ignore(1);
    ss >> c;
    ss.ignore(1);
    ss >> opFmt16;
    ss.ignore(1);
    ss >> res;
    ss.ignore(1);
    ss >> resFmt16;
    ss.ignore(1);
    ss >> flags;

    uint8_t rm = static_cast<uint8_t>(rm16);
    uint8_t opFmt = static_cast<uint8_t>(opFmt16);
    uint8_t resFmt = static_cast<uint8_t>(resFmt16);
#endif

    mp::uint128_t newRes;
    uint8_t newFlags;
    MPIntermResult intermRes;

    // Call reference model

    int success = reference_model(op, rm, a, b, c, opFmt, resFmt, newRes, &newFlags, intermRes);

    if (success == EXIT_FAILURE) {
        return "";
        // return EXIT_FAILURE;
    }

    // newRes += newRes128.upper;
    // newRes <<= 64;
    // newRes += newRes128.lower;

    // char output[512];

    std::stringstream output;
    output << std::hex << std::setfill('0');
    output << std::setw(8) << op << '_';
    output << std::setw(2) << rm16 << '_';
    output << std::setw(32) << a << '_' << std::setw(32) << b << '_' << std::setw(32) << c << '_';
    output << std::setw(2) << opFmt16 << '_';
    output << std::setw(32) << newRes << '_';
    output << std::setw(2) << resFmt16 << '_' << std::setw(2) << static_cast<uint16_t>(newFlags) << '_';
    output << std::setw(1) << intermRes.sign << '_';
    output << std::setw(8) << intermRes.exp << '_';
    output << std::setw(64) << intermRes.sig << '_';
    output << std::setw(64) << intermRes.fma_pre_addition;

    output << "\n";

    // snprintf(
    //     output,
    //     512,
    //     "%08x_%02x_%016llx%016llx_%016llx%016llx_%016llx%016llx_%02x_%016llx%016llx_%02x_%02x_%01x_%08x_%016llx%"
    //     "016llx%016llx\n",
    //     op,
    //     rm,
    //     a128.upper,
    //     a128.lower,
    //     b128.upper,
    //     b128.lower,
    //     c128.upper,
    //     c128.lower,
    //     opFmt,
    //     newRes.upper,
    //     newRes.lower,
    //     resFmt,
    //     newFlags,
    //     intermRes.sign,
    //     intermRes.exp,
    //     intermRes.sig64,
    //     intermRes.sig0,
    //     intermRes.sigExtra
    // );

    if (!suppress_error_check) {
        if (res != newRes || flags != newFlags) { // Outputs or flags do not match
            // res128.upper != newRes.upper || res128.lower != newRes.lower || // outputs don't match
            // flags != newFlags) {                                            // flags   don't match
            output = std::stringstream();
            output << "Error: testvector output doesn't match expected value\nTestVector output: ";
            output << std::hex << std::setw(32) << res;
            output << "\nExpected output: ";
            output << newRes;
            output << std::setw(2) << "\nTestVector Flags: " << flags << "\nExpected Flags: " << newFlags
                   << "\nOperation: " << op << "\n";
            // snprintf(
            //     output,
            //     512,
            //     "Error: testvector output doesn't match expected value\nTestVector output: %016llx%016llx\nExpected "
            //     "output:   %016llx%016llx\nTestVector Flags: %02x\nExpected Flags: %02x\nOperation: %08x\n",
            //     res128.upper,
            //     res128.lower,
            //     newRes128.upper,
            //     newRes128.lower,
            //     flags,
            //     newFlags,
            //     op
            // );

            return output.str();
        }
    }

    return output.str();
}
