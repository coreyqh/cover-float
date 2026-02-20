covergroup B3_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    // TODO: would be nicer as parameters such as
    //       [INTERM_M_BITS - F32_M_BITS : 0]
    f16_LSB:  coverpoint CFI.intermM[182] {
        type_option.weight = 0;
    }
    f32_LSB:  coverpoint CFI.intermM[169] {
        type_option.weight = 0;
    }
    f64_LSB:  coverpoint CFI.intermM[140] {
        type_option.weight = 0;
    }
    f128_LSB: coverpoint CFI.intermM[80]  {
        type_option.weight = 0;
    }
    bf16_LSB: coverpoint CFI.intermM[182] {
        type_option.weight = 0;
    }

    f16_guard:  coverpoint CFI.intermM[183] {
        type_option.weight = 0;
    }
    f32_guard:  coverpoint CFI.intermM[170] {
        type_option.weight = 0;
    }
    f64_guard:  coverpoint CFI.intermM[141] {
        type_option.weight = 0;
    }
    f128_guard: coverpoint CFI.intermM[81]  {
        type_option.weight = 0;
    }
    bf16_guard: coverpoint CFI.intermM[183] {
        type_option.weight = 0;

    }

    f16_sticky:  coverpoint |CFI.intermM[184:0] {
        type_option.weight = 0;
    }
    f32_sticky:  coverpoint |CFI.intermM[171:0] {
        type_option.weight = 0;
    }
    f64_sticky:  coverpoint |CFI.intermM[142:0] {
        type_option.weight = 0;
    }
    f128_sticky: coverpoint |CFI.intermM[82:0]  {
        type_option.weight = 0;
    }
    bf16_sticky: coverpoint |CFI.intermM[184:0] {
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
        B3_f32:  cross F32_result_fmt, op_arith_conv, rounding_mode_all, f16_LSB,  f16_guard,  f16_sticky;
    `endif // COVER_F32

    `ifdef COVER_F64
        B3_f64:  cross F64_result_fmt, op_arith_conv, rounding_mode_all, f32_LSB,  f32_guard,  f32_sticky;
    `endif // COVER_F64

    `ifdef COVER_F16
        B3_f16:  cross F16_result_fmt, op_arith_conv, rounding_mode_all, f64_LSB,  f64_guard,  f64_sticky;
    `endif // COVER_F16

    `ifdef COVER_BF16
        B3_bf16: cross BF16_result_fmt, op_arith_conv, rounding_mode_all, f128_LSB, f128_guard, f128_sticky;
    `endif // COVER_BF16

    `ifdef COVER_F128
        B3_f128: cross F128_result_fmt, op_arith_conv, rounding_mode_all, bf16_LSB, bf16_guard, bf16_sticky;
    `endif // COVER_F128


endgroup
