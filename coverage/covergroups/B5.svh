// Copyright (C) 2025-26 Harvey Mudd College
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, any work distributed under the
// License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions
// and limitations under the License.

covergroup B5_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    /************************************************************************
    General Helper Coverpoints
    ************************************************************************/

    FP_result_ops: coverpoint CFI.op {
        type_option.weight = 0;
        bins mul    = {OP_MUL};
        bins div    = {OP_DIV};
        bins fmadd  = {OP_FMADD};
        bins fmsub  = {OP_FMSUB};
        bins fnmadd = {OP_FNMADD};
        bins fnmsub = {OP_FNMSUB};
}

    rounding_mode_all: coverpoint CFI.rm {
        type_option.weight = 0;
        bins round_near_even   = {ROUND_NEAR_EVEN};
        bins round_minmag      = {ROUND_MINMAG};
        bins round_min         = {ROUND_MIN};
        bins round_max         = {ROUND_MAX};
        bins round_near_maxmag = {ROUND_NEAR_MAXMAG};
    }

    F16_result_fmt: coverpoint (CFI.resultFmt == FMT_HALF) {
        type_option.weight = 0;
        bins f16 = {1};
    }

    BF16_result_fmt: coverpoint (CFI.resultFmt == FMT_BF16) {
        type_option.weight = 0;
        bins bf16 = {1};
    }

    F32_result_fmt: coverpoint (CFI.resultFmt == FMT_SINGLE) {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F64_result_fmt: coverpoint (CFI.resultFmt == FMT_DOUBLE) {
        type_option.weight = 0;
        bins f64 = {1};
    }

    F128_result_fmt: coverpoint (CFI.resultFmt == FMT_QUAD) {
        type_option.weight = 0;
        bins f128 = {1};
    }

    /************************************************************************
    Underflow Boundary Helper Coverpoints
    ************************************************************************/

    // cases i & ii
    FP_subnorm: coverpoint (CFI.intermX == 0 && CFI.intermM != 0) {
        type_option.weight = 0;

        bins subnorm = {1};
    }

    // cases iii & iv

    //                                          Guard bit                                       sticky bit
    F32_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F32_M_BITS] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F32_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F32_M_BITS] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F64_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F64_M_BITS] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F64_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F64_M_BITS] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F128_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F128_M_BITS] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F128_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F128_M_BITS] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F16_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F16_M_BITS] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F16_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F16_M_BITS] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    BF16_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: BF16_M_BITS] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    BF16_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: BF16_M_BITS] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    // cases v & vi

    //                                          Guard bit                                       sticky bit
    F32_minNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 2) : 0]}
    //   implicit leading 1 (norm)           all zero fraction (except for Guard and sticky)
        iff (CFI.intermX != 0 && CFI.intermM[INTERM_M_BITS -1 -: F32_M_BITS] == 0) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F32_minNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all ones fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -1 -: F32_M_BITS] == '1) {
            type_option.weight = 0;

            bins minNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F64_minNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 2) : 0]}
    //   implicit leading 1 (norm)           all zero fraction (except for Guard and sticky)
        iff (CFI.intermX != 0 && CFI.intermM[INTERM_M_BITS -1 -: F64_M_BITS] == 0) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F64_minNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all ones fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -1 -: F64_M_BITS] == '1) {
            type_option.weight = 0;

            bins minNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F128_minNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 2) : 0]}
    //   implicit leading 1 (norm)           all zero fraction (except for Guard and sticky)
        iff (CFI.intermX != 0 && CFI.intermM[INTERM_M_BITS -1 -: F128_M_BITS] == 0) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F128_minNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all ones fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -1 -: F128_M_BITS] == '1) {
            type_option.weight = 0;

            bins minNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F16_minNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 2) : 0]}
    //   implicit leading 1 (norm)           all zero fraction (except for Guard and sticky)
        iff (CFI.intermX != 0 && CFI.intermM[INTERM_M_BITS -1 -: F16_M_BITS] == 0) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F16_minNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all ones fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -1 -: F16_M_BITS] == '1) {
            type_option.weight = 0;

            bins minNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    BF16_minNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 2) : 0]}
    //   implicit leading 1 (norm)           all zero fraction (except for Guard and sticky)
        iff (CFI.intermX != 0 && CFI.intermM[INTERM_M_BITS -1 -: BF16_M_BITS] == 0) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    BF16_minNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all ones fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -1 -: BF16_M_BITS] == '1) {
            type_option.weight = 0;

            bins minNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }


    // cases vii & viii
    F32_btw_minSubnorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubnorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F32_M_BITS)) - 1)]};
    }

    F64_btw_minSubnorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubnorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F64_M_BITS)) - 1)]};
    }

    F128_btw_minSubnorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubnorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F128_M_BITS)) - 1)]};
    }

    F16_btw_minSubnorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubnorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F16_M_BITS)) - 1)]};
    }

    BF16_btw_minSubnorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubnorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - BF16_M_BITS)) - 1)]};
    }

    // case ix
    FP_minNorm_p5_exp_range: coverpoint CFI.intermX {
        type_option.weight = 0;

        // minnorm.exp is 1 (unbiased) regardless of precision, so this covers the range [minnorm.exp , minnorm.exp + 5]
        bins exp_range[] = {[1:6]};
    }

    /************************************************************************
    Main Coverpoints
    ************************************************************************/

// TODO: Missing certain ignore bins
// TODO: Missing narrowing converts

    `ifdef COVER_F32
        B5_F32_subnorm:              cross FP_result_ops, rounding_mode_all, FP_subnorm,              F32_result_fmt;
        B5_F32_minSubnorm_p_3ulp:    cross FP_result_ops, rounding_mode_all, F32_minSubnorm_p_3ulp,   F32_result_fmt;
        B5_F32_minSubnorm_m_3ulp:    cross FP_result_ops, rounding_mode_all, F32_minSubnorm_m_3ulp,   F32_result_fmt;
        B5_F32_minNorm_p_3ulp:       cross FP_result_ops, rounding_mode_all, F32_minNorm_p_3ulp,      F32_result_fmt;
        B5_F32_minNorm_m_3ulp:       cross FP_result_ops, rounding_mode_all, F32_minNorm_m_3ulp,      F32_result_fmt;
        B5_F32_btw_minSubnorm_zero:  cross FP_result_ops, rounding_mode_all, F32_btw_minSubnorm_zero, F32_result_fmt;
        B5_F32_minNorm_p5_exp_range: cross FP_result_ops, rounding_mode_all, FP_minNorm_p5_exp_range, F32_result_fmt;
    `endif

    `ifdef COVER_F64
        B5_F64_subnorm:              cross FP_result_ops, rounding_mode_all, FP_subnorm,              F64_result_fmt;
        B5_F64_minSubnorm_p_3ulp:    cross FP_result_ops, rounding_mode_all, F64_minSubnorm_p_3ulp,   F64_result_fmt;
        B5_F64_minSubnorm_m_3ulp:    cross FP_result_ops, rounding_mode_all, F64_minSubnorm_m_3ulp,   F64_result_fmt;
        B5_F64_minNorm_p_3ulp:       cross FP_result_ops, rounding_mode_all, F64_minNorm_p_3ulp,      F64_result_fmt;
        B5_F64_minNorm_m_3ulp:       cross FP_result_ops, rounding_mode_all, F64_minNorm_m_3ulp,      F64_result_fmt;
        B5_F64_btw_minSubnorm_zero:  cross FP_result_ops, rounding_mode_all, F64_btw_minSubnorm_zero, F64_result_fmt;
        B5_F64_minNorm_p5_exp_range: cross FP_result_ops, rounding_mode_all, FP_minNorm_p5_exp_range, F64_result_fmt;
    `endif

    `ifdef COVER_F128
        B5_F128_subnorm:              cross FP_result_ops, rounding_mode_all, FP_subnorm,               F128_result_fmt;
        B5_F128_minSubnorm_p_3ulp:    cross FP_result_ops, rounding_mode_all, F128_minSubnorm_p_3ulp,   F128_result_fmt;
        B5_F128_minSubnorm_m_3ulp:    cross FP_result_ops, rounding_mode_all, F128_minSubnorm_m_3ulp,   F128_result_fmt;
        B5_F128_minNorm_p_3ulp:       cross FP_result_ops, rounding_mode_all, F128_minNorm_p_3ulp,      F128_result_fmt;
        B5_F128_minNorm_m_3ulp:       cross FP_result_ops, rounding_mode_all, F128_minNorm_m_3ulp,      F128_result_fmt;
        B5_F128_btw_minSubnorm_zero:  cross FP_result_ops, rounding_mode_all, F128_btw_minSubnorm_zero, F128_result_fmt;
        B5_F128_minNorm_p5_exp_range: cross FP_result_ops, rounding_mode_all, FP_minNorm_p5_exp_range,  F128_result_fmt;
    `endif

    `ifdef COVER_F16
        B5_F16_subnorm:              cross FP_result_ops, rounding_mode_all, FP_subnorm,              F16_result_fmt;
        B5_F16_minSubnorm_p_3ulp:    cross FP_result_ops, rounding_mode_all, F16_minSubnorm_p_3ulp,   F16_result_fmt;
        B5_F16_minSubnorm_m_3ulp:    cross FP_result_ops, rounding_mode_all, F16_minSubnorm_m_3ulp,   F16_result_fmt;
        B5_F16_minNorm_p_3ulp:       cross FP_result_ops, rounding_mode_all, F16_minNorm_p_3ulp,      F16_result_fmt;
        B5_F16_minNorm_m_3ulp:       cross FP_result_ops, rounding_mode_all, F16_minNorm_m_3ulp,      F16_result_fmt;
        B5_F16_btw_minSubnorm_zero:  cross FP_result_ops, rounding_mode_all, F16_btw_minSubnorm_zero, F16_result_fmt;
        B5_F16_minNorm_p5_exp_range: cross FP_result_ops, rounding_mode_all, FP_minNorm_p5_exp_range, F16_result_fmt;
    `endif

    `ifdef COVER_BF16
        B5_BF16_subnorm:              cross FP_result_ops, rounding_mode_all, FP_subnorm,               BF16_result_fmt;
        B5_BF16_minSubnorm_p_3ulp:    cross FP_result_ops, rounding_mode_all, BF16_minSubnorm_p_3ulp,   BF16_result_fmt;
        B5_BF16_minSubnorm_m_3ulp:    cross FP_result_ops, rounding_mode_all, BF16_minSubnorm_m_3ulp,   BF16_result_fmt;
        B5_BF16_minNorm_p_3ulp:       cross FP_result_ops, rounding_mode_all, BF16_minNorm_p_3ulp,      BF16_result_fmt;
        B5_BF16_minNorm_m_3ulp:       cross FP_result_ops, rounding_mode_all, BF16_minNorm_m_3ulp,      BF16_result_fmt;
        B5_BF16_btw_minSubnorm_zero:  cross FP_result_ops, rounding_mode_all, BF16_btw_minSubnorm_zero, BF16_result_fmt;
        B5_BF16_minNorm_p5_exp_range: cross FP_result_ops, rounding_mode_all, FP_minNorm_p5_exp_range,  BF16_result_fmt;
    `endif

endgroup
