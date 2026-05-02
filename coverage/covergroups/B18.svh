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

// Ryan Wolk (rwolk@g.hmc.edu)

covergroup B18_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    // Source Format Helpers

    F16_src_fmt: coverpoint (CFI.operandFmt == FMT_HALF) {
        type_option.weight = 0;
        bins f16 = {1};
    }

    BF16_src_fmt: coverpoint (CFI.operandFmt == FMT_BF16) {
        type_option.weight = 0;
        bins bf16 = {1};
    }

    F32_src_fmt: coverpoint (CFI.operandFmt == FMT_SINGLE) {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F64_src_fmt: coverpoint (CFI.operandFmt == FMT_DOUBLE) {
        type_option.weight = 0;
        bins f64 = {1};
    }

    F128_src_fmt: coverpoint (CFI.operandFmt == FMT_QUAD) {
        type_option.weight = 0;
        bins f128 = {1};
    }

    // FMA Instructions

    FMA_ops: coverpoint CFI.op {
        type_option.weight = 0;
        bins fmadd = { OP_FMADD };
        bins fmsub = { OP_FMSUB };
        bins fnmadd = { OP_FNMADD };
        bins fnmsub = { OP_FNMSUB };
    }

    /******************************************************************
     * Case I: Rounding Results
     ******************************************************************/

    // The FMA Pre-Addition Result is of the form 2.2nf, so here there are
    // 2.(23 bits of mantissa) (guard: nf-1) (sticky: nf-2 --> 0)
    F32_product_lsb: coverpoint |CFI.fmaPreAddition[F32_M_BITS] {
        type_option.weight = 0;
        bins lsb0 = { 0 };
        bins lsb1 = { 1 };
    }
    F32_product_guard: coverpoint CFI.fmaPreAddition[F32_M_BITS-1] {
        type_option.weight = 0;
        bins guard0 = { 0 };
        bins guard1 = { 1 };
    }
    F32_product_sticky: coverpoint |CFI.fmaPreAddition[F32_M_BITS-2:0] {
        type_option.weight = 0;
        bins sticky0 = { 0 };
        bins sticky1 = { 1 };
    }
    F32_interm_guard_zero:  coverpoint CFI.intermM[INTERM_M_BITS - F32_M_BITS - 1] {
        type_option.weight = 0;
        bins zero = { 0 };
    }
    F32_interm_sticky_zero:  coverpoint |CFI.intermM[INTERM_M_BITS - F32_M_BITS - 2 : 0] {
        type_option.weight = 0;
        bins zero = { 0 };
    }

    F64_product_lsb: coverpoint |CFI.fmaPreAddition[F64_M_BITS] {
        type_option.weight = 0;
        bins lsb0 = { 0 };
        bins lsb1 = { 1 };
    }
    F64_product_guard: coverpoint CFI.fmaPreAddition[F64_M_BITS-1] {
        type_option.weight = 0;
        bins guard0 = { 0 };
        bins guard1 = { 1 };
    }
    F64_product_sticky: coverpoint |CFI.fmaPreAddition[F64_M_BITS-2:0] {
        type_option.weight = 0;
        bins sticky0 = { 0 };
        bins sticky1 = { 1 };
    }
    F64_interm_guard_zero:  coverpoint CFI.intermM[INTERM_M_BITS - F64_M_BITS - 1] {
        type_option.weight = 0;
        bins zero = { 0 };
    }
    F64_interm_sticky_zero:  coverpoint |CFI.intermM[INTERM_M_BITS - F64_M_BITS - 2 : 0] {
        type_option.weight = 0;
        bins zero = { 0 };
    }

    F128_product_lsb: coverpoint |CFI.fmaPreAddition[F128_M_BITS] {
        type_option.weight = 0;
        bins lsb0 = { 0 };
        bins lsb1 = { 1 };
    }
    F128_product_guard: coverpoint CFI.fmaPreAddition[F128_M_BITS-1] {
        type_option.weight = 0;
        bins guard0 = { 0 };
        bins guard1 = { 1 };
    }
    F128_product_sticky: coverpoint |CFI.fmaPreAddition[F128_M_BITS-2:0] {
        type_option.weight = 0;
        bins sticky0 = { 0 };
        bins sticky1 = { 1 };
    }
    F128_interm_guard_zero:  coverpoint CFI.intermM[INTERM_M_BITS - F128_M_BITS - 1] {
        type_option.weight = 0;
        bins zero = { 0 };
    }
    F128_interm_sticky_zero:  coverpoint |CFI.intermM[INTERM_M_BITS - F128_M_BITS - 2 : 0] {
        type_option.weight = 0;
        bins zero = { 0 };
    }

    F16_product_lsb: coverpoint |CFI.fmaPreAddition[F16_M_BITS] {
        type_option.weight = 0;
        bins lsb0 = { 0 };
        bins lsb1 = { 1 };
    }
    F16_product_guard: coverpoint CFI.fmaPreAddition[F16_M_BITS-1] {
        type_option.weight = 0;
        bins guard0 = { 0 };
        bins guard1 = { 1 };
    }
    F16_product_sticky: coverpoint |CFI.fmaPreAddition[F16_M_BITS-2:0] {
        type_option.weight = 0;
        bins sticky0 = { 0 };
        bins sticky1 = { 1 };
    }
    F16_interm_guard_zero:  coverpoint CFI.intermM[INTERM_M_BITS - F16_M_BITS - 1] {
        type_option.weight = 0;
        bins zero = { 0 };
    }
    F16_interm_sticky_zero:  coverpoint |CFI.intermM[INTERM_M_BITS - F16_M_BITS - 2 : 0] {
        type_option.weight = 0;
        bins zero = { 0 };
    }

    BF16_product_lsb: coverpoint |CFI.fmaPreAddition[BF16_M_BITS] {
        type_option.weight = 0;
        bins lsb0 = { 0 };
        bins lsb1 = { 1 };
    }
    BF16_product_guard: coverpoint CFI.fmaPreAddition[BF16_M_BITS-1] {
        type_option.weight = 0;
        bins guard0 = { 0 };
        bins guard1 = { 1 };
    }
    BF16_product_sticky: coverpoint |CFI.fmaPreAddition[BF16_M_BITS-2:0] {
        type_option.weight = 0;
        bins sticky0 = { 0 };
        bins sticky1 = { 1 };
    }
    BF16_interm_guard_zero:  coverpoint CFI.intermM[INTERM_M_BITS - BF16_M_BITS - 1] {
        type_option.weight = 0;
        bins zero = { 0 };
    }
    BF16_interm_sticky_zero:  coverpoint |CFI.intermM[INTERM_M_BITS - BF16_M_BITS - 2 : 0] {
        type_option.weight = 0;
        bins zero = { 0 };
    }

    // Useful coverpoint for both case ii and case iii
    F32_sign: coverpoint CFI.result[31] {
        type_option.weight = 0;
        bins pos = {0};
        bins neg = {1};
    }

    F64_sign: coverpoint CFI.result[63] {
        type_option.weight = 0;
        bins pos = {0};
        bins neg = {1};
    }

    F128_sign: coverpoint CFI.result[127] {
        type_option.weight = 0;
        bins pos = {0};
        bins neg = {1};
    }

    F16_sign: coverpoint CFI.result[15] {
        type_option.weight = 0;
        bins pos = {0};
        bins neg = {1};
    }

    BF16_sign: coverpoint CFI.result[15] {
        type_option.weight = 0;
        bins pos = {0};
        bins neg = {1};
    }

    /************************************************************************
    Underflow Boundary Helper Coverpoints (from B4, written by Corey Hickson)
    ************************************************************************/

    // cases i & ii
    F32_maxNorm_pm_3ulp: coverpoint CFI.intermM[(INTERM_M_BITS - F32_M_BITS) -: 3]
        iff (CFI.intermX == F32_MAXNORM_EXP && CFI.intermM[(INTERM_M_BITS - 1) -: F32_M_BITS - 1] == '1) {
            type_option.weight = 0;

            bins maxNorm_pm_3ulp[] = {[3'b001 : 3'b111]};
    }

    F64_maxNorm_pm_3ulp: coverpoint CFI.intermM[(INTERM_M_BITS - F64_M_BITS) -: 3]
        iff (CFI.intermX == F64_MAXNORM_EXP && CFI.intermM[(INTERM_M_BITS - 1) -: F64_M_BITS - 1] == '1) {
            type_option.weight = 0;

            bins maxNorm_pm_3ulp[] = {[3'b001 : 3'b111]};
    }

    F128_maxNorm_pm_3ulp: coverpoint CFI.intermM[(INTERM_M_BITS - F128_M_BITS) -: 3]
        iff (CFI.intermX == F128_MAXNORM_EXP && CFI.intermM[(INTERM_M_BITS - 1) -: F128_M_BITS - 1] == '1) {
            type_option.weight = 0;

            bins maxNorm_pm_3ulp[] = {[3'b001 : 3'b111]};
    }

    F16_maxNorm_pm_3ulp: coverpoint CFI.intermM[(INTERM_M_BITS - F16_M_BITS) -: 3]
        iff (CFI.intermX == F16_MAXNORM_EXP && CFI.intermM[(INTERM_M_BITS - 1) -: F16_M_BITS - 1] == '1) {
            type_option.weight = 0;

            bins maxNorm_pm_3ulp[] = {[3'b001 : 3'b111]};
    }

    BF16_maxNorm_pm_3ulp: coverpoint CFI.intermM[(INTERM_M_BITS - BF16_M_BITS) -: 3]
        iff (CFI.intermX == BF16_MAXNORM_EXP && CFI.intermM[(INTERM_M_BITS - 1) -: BF16_M_BITS - 1] == '1) {
            type_option.weight = 0;

            bins maxNorm_pm_3ulp[] = {[3'b001 : 3'b111]};
    }

    // cases vii & viii
    F32_gt_maxNorm_p_3ulp: coverpoint CFI.intermM iff (CFI.intermX == F32_MAXNORM_EXP) {
        type_option.weight = 0;

        bins gt_maxNorm = {[ ('1 << (INTERM_M_BITS - F32_M_BITS - 2)) : $]};
    }

    F64_gt_maxNorm_p_3ulp: coverpoint CFI.intermM iff (CFI.intermX == F64_MAXNORM_EXP) {
        type_option.weight = 0;

        bins gt_maxNorm = {[ ('1 << (INTERM_M_BITS - F64_M_BITS - 2)) : $]};
    }

    F128_gt_maxNorm_p_3ulp: coverpoint CFI.intermM iff (CFI.intermX == F128_MAXNORM_EXP) {
        type_option.weight = 0;

        bins gt_maxNorm = {[ ('1 << (INTERM_M_BITS - F128_M_BITS - 2)) : $]};
    }

    F16_gt_maxNorm_p_3ulp: coverpoint CFI.intermM iff (CFI.intermX == F16_MAXNORM_EXP) {
        type_option.weight = 0;

        bins gt_maxNorm = {[ ('1 << (INTERM_M_BITS - F16_M_BITS - 2)) : $]};
    }

    BF16_gt_maxNorm_p_3ulp: coverpoint CFI.intermM iff (CFI.intermX == BF16_MAXNORM_EXP) {
        type_option.weight = 0;

        bins gt_maxNorm = {[ ('1 << (INTERM_M_BITS - BF16_M_BITS - 2)) : $]};
    }

    // case v
    F32_maxNorm_pm3_exp_range: coverpoint CFI.intermX {
        type_option.weight = 0;

        bins exp_range[] = {[ F32_MAXNORM_EXP - 3 : F32_MAXNORM_EXP + 3 ]};
    }
    F64_maxNorm_pm3_exp_range: coverpoint CFI.intermX {
        type_option.weight = 0;

        bins exp_range[] = {[ F64_MAXNORM_EXP - 3 : F64_MAXNORM_EXP + 3 ]};
    }
    F128_maxNorm_pm3_exp_range: coverpoint CFI.intermX {
        type_option.weight = 0;

        bins exp_range[] = {[ F128_MAXNORM_EXP - 3 : F128_MAXNORM_EXP + 3 ]};
    }
    F16_maxNorm_pm3_exp_range: coverpoint CFI.intermX {
        type_option.weight = 0;

        bins exp_range[] = {[ F16_MAXNORM_EXP - 3 : F16_MAXNORM_EXP + 3 ]};
    }
    BF16_maxNorm_pm3_exp_range: coverpoint CFI.intermX {
        type_option.weight = 0;

        bins exp_range[] = {[ BF16_MAXNORM_EXP - 3 : BF16_MAXNORM_EXP + 3 ]};
    }


    /************************************************************************
    Underflow Boundary Helper Coverpoints (from B5, commit: f5a2369 by Corey Hickson)
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
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F32_M_BITS +1] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F32_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F32_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F32_M_BITS +1] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F64_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F64_M_BITS +1] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F64_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F64_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F64_M_BITS +1] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F128_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F128_M_BITS +1] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F128_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F128_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F128_M_BITS +1] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F16_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F16_M_BITS +1] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    F16_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - F16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: F16_M_BITS +1] == 0) {
            type_option.weight = 0;

            bins minSubNorm_m_3ulp[] = {[2'b01 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    BF16_minSubNorm_p_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           single 1 in LSB (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: BF16_M_BITS +1] == 1) {
            type_option.weight = 0;

            bins minNorm_p_3ulp[] = {[2'b00 : 2'b11]};
    }

    //                                          Guard bit                                       sticky bit
    BF16_minSubNorm_m_3ulp: coverpoint {CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 1)], |CFI.intermM[(INTERM_M_BITS - BF16_M_BITS - 2) : 0]}
    //   implicit leading 0 (subnorm)           all zeros fraction (except for Guard and sticky)
        iff (CFI.intermX == 0 && CFI.intermM[INTERM_M_BITS -: BF16_M_BITS +1] == 0) {
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
    F32_btw_minSubNorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubNorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F32_M_BITS)) - 1)]};
    }

    F64_btw_minSubNorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubNorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F64_M_BITS)) - 1)]};
    }

    F128_btw_minSubNorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubNorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F128_M_BITS)) - 1)]};
    }

    F16_btw_minSubNorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubNorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - F16_M_BITS)) - 1)]};
    }

    BF16_btw_minSubNorm_zero: coverpoint CFI.intermM iff (CFI.intermX == 0) {
        type_option.weight = 0;

        // shift 1 into the ULP position, subtract one to be in the exclusive range (0 , minSubNorm)
        bins btw_minSubNorm_zero = {[1 : ((INTERM_M_BITS'(1) << (INTERM_M_BITS - BF16_M_BITS)) - 1)]};
    }

    // case ix
    FP_minNorm_p5_exp_range: coverpoint CFI.intermX {
        type_option.weight = 0;

        // minnorm.exp is 1 (unbiased) regardless of precision, so this covers the range [minnorm.exp , minnorm.exp + 5]
        bins exp_range[] = {[1:6]};
    }

    `ifdef COVER_F32
        B18_case_i_f32: cross F32_src_fmt, FMA_ops, F32_product_lsb, F32_product_guard, F32_product_sticky, F32_interm_guard_zero, F32_interm_sticky_zero;

        B18_case_ii_b4_maxNorm_pm_3ulp_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_maxNorm_pm_3ulp;
        B18_case_ii_b4_gt_maxNorm_p_3ulp_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_gt_maxNorm_p_3ulp;
        B18_case_ii_b4_maxNorm_pm3_exp_range_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_maxNorm_pm3_exp_range;

        B18_case_iii_b5_subnorm_f32: cross F32_src_fmt, FMA_ops, F32_sign, FP_subnorm;
        B18_case_iii_b5_minSubNorm_p_3ulp_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_minSubNorm_p_3ulp;
        B18_case_iii_b5_minSubNorm_m_3ulp_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_minSubNorm_m_3ulp;
        B18_case_iii_b5_minNorm_p_3ulp_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_minNorm_p_3ulp;
        B18_case_iii_b5_minNorm_m_3ulp_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_minNorm_m_3ulp;
        B18_case_iii_b5_btw_minSubNorm_zero_f32: cross F32_src_fmt, FMA_ops, F32_sign, F32_btw_minSubNorm_zero;
        B18_case_iii_b5_minNorm_p5_exp_range_f32: cross F32_src_fmt, FMA_ops, FP_minNorm_p5_exp_range; // No Sign in Aharoni et al
    `endif

    `ifdef COVER_F64
        B18_case_i_f64: cross F64_src_fmt, FMA_ops, F64_product_lsb, F64_product_guard, F64_product_sticky, F64_interm_guard_zero, F64_interm_sticky_zero;

        B18_case_ii_b4_maxNorm_pm_3ulp_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_maxNorm_pm_3ulp;
        B18_case_ii_b4_gt_maxNorm_p_3ulp_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_gt_maxNorm_p_3ulp;
        B18_case_ii_b4_maxNorm_pm3_exp_range_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_maxNorm_pm3_exp_range;

        B18_case_iii_b5_subnorm_f64: cross F64_src_fmt, FMA_ops, F64_sign, FP_subnorm;
        B18_case_iii_b5_minSubNorm_p_3ulp_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_minSubNorm_p_3ulp;
        B18_case_iii_b5_minSubNorm_m_3ulp_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_minSubNorm_m_3ulp;
        B18_case_iii_b5_minNorm_p_3ulp_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_minNorm_p_3ulp;
        B18_case_iii_b5_minNorm_m_3ulp_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_minNorm_m_3ulp;
        B18_case_iii_b5_btw_minSubNorm_zero_f64: cross F64_src_fmt, FMA_ops, F64_sign, F64_btw_minSubNorm_zero;
        B18_case_iii_b5_minNorm_p5_exp_range_f64: cross F64_src_fmt, FMA_ops, FP_minNorm_p5_exp_range; // No Sign in Aharoni et al
    `endif

    `ifdef COVER_F128
        B18_case_i_f128: cross F128_src_fmt, FMA_ops, F128_product_lsb, F128_product_guard, F128_product_sticky, F128_interm_guard_zero, F128_interm_sticky_zero;

        B18_case_ii_b4_maxNorm_pm_3ulp_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_maxNorm_pm_3ulp;
        B18_case_ii_b4_gt_maxNorm_p_3ulp_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_gt_maxNorm_p_3ulp;
        B18_case_ii_b4_maxNorm_pm3_exp_range_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_maxNorm_pm3_exp_range;

        B18_case_iii_b5_subnorm_f128: cross F128_src_fmt, FMA_ops, F128_sign, FP_subnorm;
        B18_case_iii_b5_minSubNorm_p_3ulp_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_minSubNorm_p_3ulp;
        B18_case_iii_b5_minSubNorm_m_3ulp_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_minSubNorm_m_3ulp;
        B18_case_iii_b5_minNorm_p_3ulp_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_minNorm_p_3ulp;
        B18_case_iii_b5_minNorm_m_3ulp_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_minNorm_m_3ulp;
        B18_case_iii_b5_btw_minSubNorm_zero_f128: cross F128_src_fmt, FMA_ops, F128_sign, F128_btw_minSubNorm_zero;
        B18_case_iii_b5_minNorm_p5_exp_range_f128: cross F128_src_fmt, FMA_ops, FP_minNorm_p5_exp_range; // No Sign in Aharoni et al
    `endif

    `ifdef COVER_F16
        B18_case_i_f16: cross F16_src_fmt, FMA_ops, F16_product_lsb, F16_product_guard, F16_product_sticky, F16_interm_guard_zero, F16_interm_sticky_zero;

        B18_case_ii_b4_maxNorm_pm_3ulp_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_maxNorm_pm_3ulp;
        B18_case_ii_b4_gt_maxNorm_p_3ulp_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_gt_maxNorm_p_3ulp;
        B18_case_ii_b4_maxNorm_pm3_exp_range_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_maxNorm_pm3_exp_range;

        B18_case_iii_b5_subnorm_f16: cross F16_src_fmt, FMA_ops, F16_sign, FP_subnorm;
        B18_case_iii_b5_minSubNorm_p_3ulp_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_minSubNorm_p_3ulp;
        B18_case_iii_b5_minSubNorm_m_3ulp_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_minSubNorm_m_3ulp;
        B18_case_iii_b5_minNorm_p_3ulp_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_minNorm_p_3ulp;
        B18_case_iii_b5_minNorm_m_3ulp_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_minNorm_m_3ulp;
        B18_case_iii_b5_btw_minSubNorm_zero_f16: cross F16_src_fmt, FMA_ops, F16_sign, F16_btw_minSubNorm_zero;
        B18_case_iii_b5_minNorm_p5_exp_range_f16: cross F16_src_fmt, FMA_ops, FP_minNorm_p5_exp_range; // No Sign in Aharoni et al
    `endif

    `ifdef COVER_BF16
        B18_case_i_bf16: cross BF16_src_fmt, FMA_ops, BF16_product_lsb, BF16_product_guard, BF16_product_sticky, BF16_interm_guard_zero, BF16_interm_sticky_zero;

        B18_case_ii_b4_maxNorm_pm_3ulp_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_maxNorm_pm_3ulp;
        B18_case_ii_b4_gt_maxNorm_p_3ulp_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_gt_maxNorm_p_3ulp;
        B18_case_ii_b4_maxNorm_pm3_exp_range_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_maxNorm_pm3_exp_range;

        B18_case_iii_b5_subnorm_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, FP_subnorm;
        B18_case_iii_b5_minSubNorm_p_3ulp_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_minSubNorm_p_3ulp;
        B18_case_iii_b5_minSubNorm_m_3ulp_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_minSubNorm_m_3ulp;
        B18_case_iii_b5_minNorm_p_3ulp_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_minNorm_p_3ulp;
        B18_case_iii_b5_minNorm_m_3ulp_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_minNorm_m_3ulp;
        B18_case_iii_b5_btw_minSubNorm_zero_bf16: cross BF16_src_fmt, FMA_ops, BF16_sign, BF16_btw_minSubNorm_zero;
        B18_case_iii_b5_minNorm_p5_exp_range_bf16: cross BF16_src_fmt, FMA_ops, FP_minNorm_p5_exp_range; // No Sign in Aharoni et al
    `endif

endgroup
