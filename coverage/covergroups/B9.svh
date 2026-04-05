
covergroup B9_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    /************************************************************************
     *
     * Operation helper coverpoint (div, rem, sqrt, mul)
     *
     ************************************************************************/

    FP_B9_ops: coverpoint CFI.op {
        type_option.weight = 0;

        bins div  = {OP_DIV};
        // bins rem  = {OP_REM};
        bins mul  = {OP_MUL};

    }

    FP_sqrt_op: coverpoint CFI.op {
        type_option.weight = 0;

        bins sqrt = {OP_SQRT};
    }

    /************************************************************************
     *
     * Operand format helper coverpoints
     *
     ************************************************************************/

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


    /************************************************************************
     *
     * Special helper coverpoints
     *
     ************************************************************************/

    F16_a_special_sigs:  coverpoint (CFI.a[F16_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F16_special_sigs.svh"
    }
    BF16_a_special_sigs: coverpoint (CFI.a[BF16_M_UPPER : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_BF16_special_sigs.svh"
    }
    F32_a_special_sigs:  coverpoint (CFI.a[F32_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F32_special_sigs.svh"
    }
    F64_a_special_sigs:  coverpoint (CFI.a[F64_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F64_special_sigs.svh"
    }
    F128_a_special_sigs: coverpoint (CFI.a[F128_M_UPPER : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F128_special_sigs.svh"
    }

    F16_b_special_sigs:  coverpoint (CFI.b[F16_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F16_special_sigs.svh"
    }
    BF16_b_special_sigs: coverpoint (CFI.b[BF16_M_UPPER : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_BF16_special_sigs.svh"
    }
    F32_b_special_sigs:  coverpoint (CFI.b[F32_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F32_special_sigs.svh"
    }
    F64_b_special_sigs:  coverpoint (CFI.b[F64_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F64_special_sigs.svh"
    }
    F128_b_special_sigs: coverpoint (CFI.b[F128_M_UPPER : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B9_F128_special_sigs.svh"
    }

    /************************************************************************
     *
     * Main crosses (precision-sorted)
     *
     ************************************************************************/

    `ifdef COVER_F16
        B9_F16:      cross B9_FP_ops,   F16_a_special_sigs, F16_b_special_sigs, F16_result_fmt;
        B9_F16_sqrt: cross B9_sqrt_ops, F16_a_special_sigs,                     F16_result_fmt;
    `endif // COVER_F16

    `ifdef COVER_BF16
        B9_BF16:      cross B9_FP_ops,   BF16_a_special_sigs, BF16_b_special_sigs, BF16_result_fmt;
        B9_BF16_sqrt: cross B9_sqrt_ops, BF16_a_special_sigs,                      BF16_result_fmt;
    `endif // COVER_BF16

    `ifdef COVER_F32
        B9_F32:      cross B9_FP_ops,   F32_a_special_sigs, F32_b_special_sigs, F32_result_fmt;
        B9_F32_sqrt: cross B9_sqrt_ops, F32_a_special_sigs,                     F32_result_fmt;
    `endif // COVER_F32

    `ifdef COVER_F64
        B9_F64:      cross B9_FP_ops,   F64_a_special_sigs, F64_b_special_sigs, F64_result_fmt;
        B9_F64_sqrt: cross B9_sqrt_ops, F64_a_special_sigs,                     F64_result_fmt;
    `endif // COVER_F64

    `ifdef COVER_F128
        B9_F128:      cross B9_FP_ops,   F128_a_special_sigs, F128_b_special_sigs, F128_result_fmt;
        B9_F128_sqrt: cross B9_sqrt_ops, F128_a_special_sigs,                      F128_result_fmt;
    `endif // COVER_F128

endgroup
