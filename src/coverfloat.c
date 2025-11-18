#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "coverfloat.h"

void softFloat_clearFlags( uint_fast8_t clearMask) {
    softfloat_exceptionFlags &= ~clearMask;
}

uint_fast8_t softFloat_getFlags () {
    return softfloat_exceptionFlags;
}

void softFloat_setRoundingMode (uint_fast8_t rm) {
    softfloat_roundingMode = rm;
}

void softfloat_getIntermResults (intermResult_t * result) {

    result->sign     = softfloat_intermediateResult.sign;
    result->exp      = softfloat_intermediateResult.exp;
    result->sig64    = softfloat_intermediateResult.sig64;
    result->sig0     = softfloat_intermediateResult.sig0;
    result->sigExtra = softfloat_intermediateResult.sigExtra;

}

void softfloat_clearIntermResults () {

    softfloat_intermediateResult.sign     = 0;
    softfloat_intermediateResult.exp      = 0;
    softfloat_intermediateResult.sig64    = 0;
    softfloat_intermediateResult.sig0     = 0;
    softfloat_intermediateResult.sigExtra = 0;

}

/*
AI CODE NEEDS MODIFICATION
*/


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


void reference_model( const uint32_t       * op,
                      const uint8_t        * rm,
                      const uint128_t      * a, 
                      const uint128_t      * b, 
                      const uint128_t      * c, 
                      const uint8_t        * operandFmt, 
                      const uint8_t        * resultFmt,

                      uint128_t            * result,
                      uint8_t              * flags ,
                      intermResult_t       * intermResult ) {
    
    
    // clear flags so we get only triggered flags
    softFloat_clearFlags(0xFF);

    // clear intermediate result to avoid reporting intermediate results for results that were not rounded
    softfloat_clearIntermResults(result);

    // set rounding mode
    softFloat_setRoundingMode(*rm);

    // nested switch statements to call softfloat functions

    switch (*op) {
        case OP_ADD: {
            
            switch (*operandFmt) {
                case FMT_SINGLE: {
                    float32_t af, bf, resultf;
                    UINT128_TO_FLOAT32(af, a);
                    UINT128_TO_FLOAT32(bf, b);
                    resultf = f32_add(af, bf);
                    FLOAT32_TO_UINT128(result, resultf);
                    break;
                }

                case FMT_DOUBLE: {
                    float64_t af, bf, resultf;
                    UINT128_TO_FLOAT64(af, a);
                    UINT128_TO_FLOAT64(bf, b);
                    resultf = f64_add(af, bf);
                    FLOAT64_TO_UINT128(result, resultf);
                    break;
                }

                /* ... */

            }
            break;
        }

        case OP_SUB: {
            
            switch (*operandFmt) {
                case FMT_SINGLE: {
                    float32_t af, bf, resultf;
                    UINT128_TO_FLOAT32(af, a);
                    UINT128_TO_FLOAT32(bf, b);
                    resultf = f32_sub(af, bf);
                    FLOAT32_TO_UINT128(result, resultf);
                    break;
                }

                case FMT_DOUBLE: {
                    float64_t af, bf, resultf;
                    UINT128_TO_FLOAT64(af, a);
                    UINT128_TO_FLOAT64(bf, b);
                    resultf = f64_add(af, bf);
                    FLOAT64_TO_UINT128(result, resultf);
                    break;
                }

                /* ... */

            }

            break;
        }

        case OP_MUL: {
            
            switch (*operandFmt) {
                case FMT_SINGLE: {
                    float32_t af, bf, resultf;
                    UINT128_TO_FLOAT32(af, a);
                    UINT128_TO_FLOAT32(bf, b);
                    resultf = f32_mul(af, bf);
                    FLOAT32_TO_UINT128(result, resultf);

                    // printf("performing single precision mul!!\n");
                    // printf("int operands are: %x and %x\n", *a, *b);
                    // printf("float operands are: %x and %x\n", af.v, bf.v);
                    // printf("float result is %x\n", resultf.v);
                    // printf("int result is %032x%032x\n", result->upper, result->lower);
                    break;
                }

                case FMT_DOUBLE: {
                    float64_t af, bf, resultf;
                    UINT128_TO_FLOAT64(af, a);
                    UINT128_TO_FLOAT64(bf, b);
                    resultf = f64_mul(af, bf);
                    FLOAT64_TO_UINT128(result, resultf);
                    break;
                }

                /* ... */

            }

            break;
        }

        /* ... */

    }

    *flags = softFloat_getFlags();
    softfloat_getIntermResults(intermResult);
}

// TODO needs complete refactor...
int main(int argc, char *argv[]) {
    bool suppress_error_check = false;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <input_file|- for stdin> <output_file|- for stdout> [--no-error-check]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 4 && strcmp(argv[3], "--no-error-check") == 0) {
        suppress_error_check = true;
    }


    FILE *fin = strcmp(argv[1], "-") == 0 ? stdin : fopen(argv[1], "r");
    if (!fin) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    FILE *fout = strcmp(argv[2], "-") == 0 ? stdout : fopen(argv[2], "w");
    if (!fout) {
        perror("Error opening output file");
        if (fin != stdin) fclose(fin);
        return EXIT_FAILURE;
    }

    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), fin)) {
        // Strip newline
        line[strcspn(line, "\r\n")] = '\0';

        char     op_str[MAX_TOKEN_LEN + 1]; // plus one for space for null terminator
        char     rm_str[MAX_TOKEN_LEN + 1];
        char      a_str[MAX_TOKEN_LEN + 1];
        char      b_str[MAX_TOKEN_LEN + 1];
        char      c_str[MAX_TOKEN_LEN + 1];
        char  opFmt_str[MAX_TOKEN_LEN + 1];
        char    res_str[MAX_TOKEN_LEN + 1];
        char resFmt_str[MAX_TOKEN_LEN + 1];
        char  flags_str[MAX_TOKEN_LEN + 1];

        if (line[0] == '/' && line[1] == '/') continue;

        if (sscanf(line, "%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_]_%48[^_ \t\r\n]", 
            op_str, rm_str, a_str, b_str, c_str, opFmt_str, res_str, resFmt_str, flags_str) != 9) {
            fprintf(stderr, "Skipping malformed line: %s\n", line);
            continue;
        }


        // unpack test vector tokens into integers to pass to the reference model

        uint32_t       op        = parse_hex_128(op_str       ).lower;
        uint8_t        rm        = parse_hex_128(rm_str       ).lower;
        uint128_t      a         = parse_hex_128(a_str        )      ;
        uint128_t      b         = parse_hex_128(b_str        )      ;
        uint128_t      c         = parse_hex_128(c_str        )      ;
        uint8_t        opFmt     = parse_hex_128(opFmt_str    ).lower;
        uint8_t        resFmt    = parse_hex_128(resFmt_str   ).lower;
        uint128_t      res       = parse_hex_128(res_str      )      ;
        uint8_t        flags     = parse_hex_128(flags_str    ).lower;
        
        
        uint128_t      newRes;
        uint8_t        newFlags;
        intermResult_t intermRes;


        // Call reference model
                
        reference_model(&op,
                        &rm,
                        &a, 
                        &b, 
                        &c, 
                        &opFmt, 
                        &resFmt,

                        &newRes,
                        &newFlags,
                        &intermRes );

        // Write cover vector (append intermediate result to test vector)
        switch (resFmt) {
            // if (suppress_error_check) {
            //     line = "%08x_%02x_%016x_%016x_%016x_%016x_%016x_%016x_%02x_%016x_%016x_%02x_%02x"
            // }

            case FMT_QUAD: {
                fprintf(fout, "%08x_%02x_%016x_%016x_%016x_%016x_%016x_%016x_%02x_%016x_%016x_%02x_%02x_%01x_%08x_%016x%016x%016x\n", 
                        op, rm, a.upper, a.lower, b.upper, b.lower, c.upper, c.lower, opFmt, newRes.upper, newRes.lower, resFmt, newFlags, 
                        intermRes.sign, intermRes.exp, intermRes.sig64, intermRes.sig0, intermRes.sigExtra);
                break;
            }
            case FMT_DOUBLE: {
                fprintf(fout, "%08x_%02x_%016x_%016x_%016x_%016x_%016x_%016x_%02x_%016x_%016x_%02x_%02x_%01x_%08x_%016x%016x%016x\n", 
                        op, rm, a.upper, a.lower, b.upper, b.lower, c.upper, c.lower, opFmt, newRes.upper, newRes.lower, resFmt, newFlags, 
                        intermRes.sign, intermRes.exp, intermRes.sig64, intermRes.sig0, intermRes.sigExtra);
                break;
            }
            case FMT_SINGLE: {
                fprintf(fout, "%08x_%02x_%016x_%016x_%016x_%016x_%016x_%016x_%02x_%016x_%016x_%02x_%02x_%01x_%08x_%08x%08x%016x%016x\n", 
                        op, rm, a.upper, a.lower, b.upper, b.lower, c.upper, c.lower, opFmt, newRes.upper, newRes.lower, resFmt, newFlags, 
                        intermRes.sign, intermRes.exp, intermRes.sig64, 0x0, intermRes.sig0, intermRes.sigExtra);
                break;
            }
            case FMT_HALF: {
                fprintf(fout, "%08x_%02x_%016x_%016x_%016x_%016x_%016x_%016x_%02x_%016x_%016x_%02x_%02x_%01x_%08x_%04x%012x%016x%016x\n", 
                        op, rm, a.upper, a.lower, b.upper, b.lower, c.upper, c.lower, opFmt, newRes.upper, newRes.lower, resFmt, newFlags, 
                        intermRes.sign, intermRes.exp, intermRes.sig64, 0x0, intermRes.sig0, intermRes.sigExtra);
                break;
            }
            case FMT_BF16: {
                fprintf(fout, "%08x_%02x_%016x_%016x_%016x_%016x_%016x_%016x_%02x_%016x_%016x_%02x_%02x_%01x_%08x_%04x%012x%016x%016x\n", 
                        op, rm, a.upper, a.lower, b.upper, b.lower, c.upper, c.lower, opFmt, newRes.upper, newRes.lower, resFmt, newFlags, 
                        intermRes.sign, intermRes.exp, intermRes.sig64, 0x0, intermRes.sig0, intermRes.sigExtra);
                break;
            }
        }


        // printf("INTERM SIG IS %016x\n\n", intermRes.sig64);

        // confirm softfloat output matches testvectors
        if (!suppress_error_check) {
            if (res.upper   != newRes.upper   || res.lower   != newRes.lower ||     // outputs don't match
                flags != newFlags                                              ) {  // flags   don't match
                fprintf(stderr, "Error: testvector output doesn't match expected value\nTestVector output: %016x%016x\nExpected output: %016x%016x\nTestVector Flags: %02x\nExpected Flags: %02x\n", 
                    res.upper, res.lower, newRes.upper, newRes.lower, flags, newFlags);
                fclose(fin);
                fclose(fout);
                return EXIT_FAILURE;
            }
        } 
    }

    if (fin  != stdin)  fclose(fin);
    if (fout != stdout) fclose(fout);

    return EXIT_SUCCESS;
}
