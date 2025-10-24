#ifndef COVERFLOAT_H_INCLUDED
#define COVERFLOAT_H_INCLUDED

#include <stdint.h>
#include "../../riscv-isa-sim/softfloat/softfloat.h" // TODO: yuck fix paths

#ifdef __cplusplus
extern "C" {
#endif

/*
typedef enum {
    OP_ADD   = 1,
    OP_SUB   = 2,
    OP_MUL   = 3,
    OP_DIV   = 4,
    OP_FMA   = 5,
    OP_SQRT  = 6,
    OP_REM   = 7,
    OP_CFI   = 8,
    OP_CIF   = 9,
    OP_QC    = 10,
    OP_SC    = 11,
    OP_EQ    = 12,
    OP_CLASS = 13
} op_t;

typedef struct packed {
    bool     sign;
    uint16_t exp;
    uint16_t sig;
} intermFloat16_t;

typedef struct packed {
    bool     sign;
    uint16_t exp;
    uint32_t sig;
} intermFloat32_t;

typedef struct packed {
    bool     sign;
    uint16_t exp;
    uint64_t sig;
} intermFloat64_t;

typedef struct packed {
    bool     sign;
    uint32_t exp;
    uint64_t sig64;
    uint64_t sig0;
    uint64_t sigExtra;
} intermFloat128_t;

typedef struct packed {
    // reported by DUT
    uint32_t         op, rm;
    uint32_t         enablebools, exceptionbools;
    float128_t       a, b, c, result;
    // reported by reference
    float128_t       expectedResult;
    intermFloat128_t intermResult;
} coverfloat128_t;

typedef struct packed {
    // reported by DUT
    uint32_t        op, rm;
    uint32_t        enablebools, exceptionbools;
    float16_t       a, b, c, result;
    // reported by reference
    float16_t       expectedResult;
    intermFloat16_t intermResult;
} coverfloat16_t;

typedef struct packed {
    // reported by DUT
    uint32_t        op, rm;
    uint32_t        enablebools, exceptionbools;
    float32_t       a, b, c, result;
    // reported by reference
    float32_t       expectedResult;
    intermFloat32_t intermResult;
} coverfloat32_t;

typedef struct packed {
    // reported by DUT
    uint32_t        op, rm;
    uint32_t        enablebools, exceptionbools;
    float64_t       a, b, c, result;
    // reported by reference
    float64_t       expectedResult;
    intermFloat64_t intermResult;
} coverfloat64_t;

coverfloat16_t  coverfloat16Ref  (uint32_t op, float16_t  a, float16_t  b, float16_t  c, uint32_t rm, uint32_t enablebools); 
coverfloat32_t  coverfloat32Ref  (uint32_t op, float32_t  a, float32_t  b, float32_t  c, uint32_t rm, uint32_t enablebools);
coverfloat64_t  coverfloat64Ref  (uint32_t op, float64_t  a, float64_t  b, float64_t  c, uint32_t rm, uint32_t enablebools);
coverfloat128_t coverfloat128Ref (uint32_t op, float128_t a, float128_t b, float128_t c, uint32_t rm, uint32_t enablebools);

*/

void softFloat_clearFlags( uint_fast8_t );

uint_fast8_t softFloat_getFlags ();

void softFloat_setRoundingMode ( uint_fast8_t );

void softfloat_getIntermResults ( intermResult_t * );


#ifdef __cplusplus
}
#endif

#endif