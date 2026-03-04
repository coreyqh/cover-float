covergroup B3_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;


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


    F16_LSB:  coverpoint CFI.intermM[INTERM_M_BITS - F16_M_BITS] {
        type_option.weight = 0;
    }
    F32_LSB:  coverpoint CFI.intermM[INTERM_M_BITS - F32_M_BITS] {
        type_option.weight = 0;
    }
    F64_LSB:  coverpoint CFI.intermM[INTERM_M_BITS - F64_M_BITS] {
        type_option.weight = 0;
    }
    F128_LSB: coverpoint CFI.intermM[INTERM_M_BITS - F128_M_BITS] {
        type_option.weight = 0;
    }
    BF16_LSB: coverpoint CFI.intermM[INTERM_M_BITS - BF16_M_BITS] {
        type_option.weight = 0;
    }


    F16_guard:  coverpoint CFI.intermM[INTERM_M_BITS - F16_M_BITS - 1] {
        type_option.weight = 0;
    }
    F32_guard:  coverpoint CFI.intermM[INTERM_M_BITS - F32_M_BITS - 1] {
        type_option.weight = 0;
    }
    F64_guard:  coverpoint CFI.intermM[INTERM_M_BITS - F64_M_BITS - 1] {
        type_option.weight = 0;
    }
    F128_guard: coverpoint CFI.intermM[INTERM_M_BITS - F128_M_BITS - 1] {
        type_option.weight = 0;
    }
    BF16_guard: coverpoint CFI.intermM[INTERM_M_BITS - BF16_M_BITS - 1] {
        type_option.weight = 0;
    }


    F16_sticky:  coverpoint |CFI.intermM[INTERM_M_BITS - F16_M_BITS - 2 : 0] {
        type_option.weight = 0;
    }
    F32_sticky:  coverpoint |CFI.intermM[INTERM_M_BITS - F32_M_BITS - 2 : 0] {
        type_option.weight = 0;
    }
    F64_sticky:  coverpoint |CFI.intermM[INTERM_M_BITS - F64_M_BITS - 2 : 0] {
        type_option.weight = 0;
    }
    F128_sticky: coverpoint |CFI.intermM[INTERM_M_BITS - F128_M_BITS - 2 : 0] {
        type_option.weight = 0;
    }
    BF16_sticky: coverpoint |CFI.intermM[INTERM_M_BITS - BF16_M_BITS - 2 : 0] {
        type_option.weight = 0;
    }

    rounding_mode_all: coverpoint CFI.rm {
        type_option.weight = 0;
        bins round_near_even   = {ROUND_NEAR_EVEN};
        bins round_minmag      = {ROUND_MINMAG};
        bins round_min         = {ROUND_MIN};
        bins round_max         = {ROUND_MAX};
        bins round_near_maxmag = {ROUND_NEAR_MAXMAG};
    }

    op_arith_conv: coverpoint CFI.op {
        type_option.weight = 0;
        `include "bins_templates/arithmetic_op_bins.svh"
        `include "bins_templates/conversion_op_bins.svh"
    }

    F16_result_fmt: coverpoint CFI.resultFmt == FMT_HALF {
        type_option.weight = 0;
        // half precision format for result
        bins f16 = {1};
    }

    BF16_result_fmt: coverpoint CFI.resultFmt == FMT_BF16 {
        type_option.weight = 0;
        // bfloat16 precision format for result
        bins bf16 = {1};
    }

    F32_result_fmt: coverpoint CFI.resultFmt == FMT_SINGLE {
        type_option.weight = 0;
        // single precision format for result
        bins f32 = {1};
    }

    F64_result_fmt: coverpoint CFI.resultFmt == FMT_DOUBLE {
        type_option.weight = 0;
        // half precision format for result
        bins f64 = {1};
    }

    F128_result_fmt: coverpoint CFI.resultFmt == FMT_QUAD {
        type_option.weight = 0;
        // quad precision format for result
        bins f128 = {1};
    }

    // main coverpoints
    `ifdef COVER_F32
        B3_F32:  cross op_arith_conv, rounding_mode_all, F32_sign, F32_LSB,  F32_guard,  F32_sticky, F32_result_fmt;
    `endif // COVER_F32

    `ifdef COVER_F64
        B3_F64:  cross op_arith_conv, rounding_mode_all, F64_sign, F64_LSB,  F64_guard,  F64_sticky, F64_result_fmt;
    `endif // COVER_F64

    `ifdef COVER_F16
        B3_F16:  cross op_arith_conv, rounding_mode_all, F16_sign, F16_LSB,  F16_guard,  F16_sticky, F16_result_fmt;
    `endif // COVER_F16

    `ifdef COVER_BF16
        B3_BF16: cross op_arith_conv, rounding_mode_all, BF16_sign, BF16_LSB, BF16_guard, BF16_sticky, BF16_result_fmt;
    `endif // COVER_BF16

    `ifdef COVER_F128
        B3_F128: cross op_arith_conv, rounding_mode_all, F128_sign, F128_LSB, F128_guard, F128_sticky, F128_result_fmt;
    `endif // COVER_F128


endgroup
