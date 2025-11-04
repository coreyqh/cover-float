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

    result->sign     = softfloat_intermediateResult->sign;
    result->exp      = softfloat_intermediateResult->exp;
    result->sig64    = softfloat_intermediateResult->sig64;
    result->sig0     = softfloat_intermediateResult->sig0;
    result->sigExtra = softfloat_intermediateResult->sigExtra;

}

/*
AI CODE NEEDS MODIFICATION
*/

// Parse a hex string into a 128-bit integer (uint128_t)
uint128_t parse_hex_128(const char *hex) {
    uint128_t value = 0;
    while (*hex) {
        char c = *hex++;
        uint8_t digit = 0;

        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
        else continue; // skip non-hex chars

        value = (value << 4) | digit;
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
    
    
    // clear flags so we ger only triggered flags
    softFloat_clearFlags(0xFF);

    // set rounding mode
    softFloat_setRoundingMode(*rm);

    // nested switch statements to call softfloat functions

    switch (*op) {
        case OP_ADD: {
            
            switch (*aFmt) {
                case FMT_SINGLE: {
                    float32_t af, bf, resultf;
                    af.val = (*a << 96) >> 96;
                    bf.val = (*b << 96) >> 96;
                    resultf = f32_add(af, bf);
                    * result = resultf.val;
                    break;
                }

                case FMT_DOUBLE: {
                    float64_t af, bf, resultf;
                    af.val = (*a << 64) >> 64;
                    bf.val = (*b << 64) >> 64;
                    resultf = f64_add(af, bf);
                    * result = resultf.val;
                    break;
                }

                /* ... */

            }
            break;
        }

        case OP_SUB: {
            
            switch (*aFmt) {
                case FMT_SINGLE: {
                    float32_t af, bf, resultf;
                    af.val = (a << 96) >> 96;
                    bf.val = (b << 96) >> 96;
                    resultf = f32_sub(af, bf);
                    * result = resultf.val;
                    break;
                }

                case FMT_DOUBLE: {
                    float64_t af, bf, resultf;
                    af.val = (a << 64) >> 64;
                    bf.val = (b << 64) >> 64;
                    resultf = f64_sub(af, bf);
                    * result = resultf.val;
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
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    FILE *fout = fopen(argv[2], "w");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return EXIT_FAILURE;
    }

    char line[TEST_VECTOR_SIZE_HEX];
    while (fgets(line, sizeof(line), fin)) {
        // Strip newline
        line[strcspn(line, "\r\n")] = '\0';

        char     op_str[MAX_TOKEN_LEN];
        char     rm_str[MAX_TOKEN_LEN];
        char      a_str[MAX_TOKEN_LEN];
        char      b_str[MAX_TOKEN_LEN];
        char      c_str[MAX_TOKEN_LEN];
        char  opFmt_str[MAX_TOKEN_LEN];
        char    res_str[MAX_TOKEN_LEN];
        char resFmt_str[MAX_TOKEN_LEN];
        char  flags_str[MAX_TOKEN_LEN];

        if (sscanf(line, "%48[^_]_%48[^_]_%48s_%48[^_]_%48[^_]_%48s_%48[^_]_%48[^_]_%48s", 
            op_str, rm_str, a_str, b_str, c_str, opFmt_str, res_str, resFmt_str, flags_str) != 9) {
            fprintf(stderr, "Skipping malformed line: %s\n", line);
            continue;
        }


        // unpack test vector tokens into integers to pass to the reference model

        uint32_t       op        = parse_hex_128(op_str       );
        uint8_t        rm        = parse_hex_128(rm_str       );
        uint128_t      a         = parse_hex_128(a_str        );
        uint128_t      b         = parse_hex_128(b_str        );
        uint128_t      c         = parse_hex_128(c_str        );
        uint8_t        opFmt     = parse_hex_128(opFmt_str    );
        uint8_t        resFmt    = parse_hex_128(resFmt_str   );
        uint128_t      res       = parse_hex_128(res_str      );
        uint8_t        flags     = parse_hex_128(flags_str    );
        
        
        uint128_t      newRes;
        uint8_t        newflags;
        intermResult_t intermRes;


        // Call reference model
                
        void reference_model(&op,
                             &rm,
                             &a, 
                             &b, 
                             &c, 
                             &opFmt, 
                             &resFmt,

                             &res,
                             &flags,
                             &intermRes );

        // Write cover vector (append intermediate result to test vector)
        fprintf(fout, "%s_%04x_%032x_%064x%064x%064x\n", 
                line, intermRes.sign, intermRes.exp, intermRes.sig64, intermRes.sig0, intermRes.sigExtra);

        // confirm softfloat output matches testvectors
        if (strcmp(res_str,     newRes_str)  != 0 ||  // outputs don't match
            strcmp(flags_str, newflags_str)  != 0) {  // flags don't match
            perror("Error: testvector output doesn't match expected value\nTestVector output: %s\nExpected output: %s", output, new_output);
            fclose(fin);
            fclose(fout);
            return EXIT_FAILURE;
        }
    }

    fclose(fin);
    fclose(fout);

    return EXIT_SUCCESS;
}
