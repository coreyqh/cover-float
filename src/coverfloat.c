#include <stdint.h>
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
