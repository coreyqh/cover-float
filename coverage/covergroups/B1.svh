covergroup B1_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    FP_result_ops: coverpoint CFI.op {
        type_option.weight = 0;
        // all operations that produce (arbitrary) FP results
        `include "bins_templates/FP_result_op_bins.svh"
    }

    FP_src1_ops: coverpoint CFI.op {
        type_option.weight = 0;
        // all operations where the first operand is FP
        `include "bins_templates/FP_src1_op_bins.svh"
    }

    FP_src2_ops: coverpoint CFI.op {
        type_option.weight = 0;
        // all operations where the second operand is FP
        `include "bins_templates/FP_src2_op_bins.svh"
    }

    FP_src3_ops: coverpoint CFI.op {
        type_option.weight = 0;
        // all operations where the third operand is FP
        `include "bins_templates/FP_src3_op_bins.svh"
    }

    F32_src_fmt: coverpoint CFI.operandFmt == FMT_SINGLE {
        type_option.weight = 0;
        // single precision format for operands
        bins f32 = {1};
    }

    F32_result_fmt: coverpoint CFI.resultFmt == FMT_SINGLE {
        type_option.weight = 0;
        // single precision format for result
        bins f32 = {1};
    }

    F32_src1_basictypes: coverpoint CFI.a[31:0] {
        type_option.weight = 0;
        `include "bins_templates/F32_basic_types_bins.svh"
    }

    F32_src2_basictypes: coverpoint CFI.b[31:0] {
        type_option.weight = 0;
        `include "bins_templates/F32_basic_types_bins.svh"
    }

    F32_src3_basictypes: coverpoint CFI.c[31:0] {
        type_option.weight = 0;
        `include "bins_templates/F32_basic_types_bins.svh"
    }

    F32_result_basictypes: coverpoint CFI.result[31:0] {
        type_option.weight = 0;
        `include "bins_templates/F32_basic_types_bins.svh"
    }

    // main coverpoints

    `ifdef COVER_F32
        B1_F32_1_operands: cross FP_src1_ops,   F32_src1_basictypes,                                           F32_src_fmt;
        B1_F32_2_operands: cross FP_src2_ops,   F32_src1_basictypes, F32_src2_basictypes,                      F32_src_fmt;
        B1_F32_3_operands: cross FP_src3_ops,   F32_src1_basictypes, F32_src2_basictypes, F32_src3_basictypes, F32_src_fmt;
        B1_F32_result:     cross FP_result_ops, F32_result_basictypes,                                         F32_result_fmt;
    `endif // COVER_F32

    `ifdef COVER_F64
        B1_F64_1_operands: cross FP_src1_ops,   F64_src1_basictypes,                                           F64_src_fmt;
        B1_F64_2_operands: cross FP_src2_ops,   F64_src1_basictypes, F64_src2_basictypes,                      F64_src_fmt;
        B1_F64_3_operands: cross FP_src3_ops,   F64_src1_basictypes, F64_src2_basictypes, F64_src3_basictypes, F64_src_fmt;
        B1_F64_result:     cross FP_result_ops, F64_result_basictypes,                                         F64_result_fmt;
    `endif // COVER_F64

    `ifdef COVER_F16
        B1_F16_1_operands: cross FP_src1_ops,   F16_src1_basictypes,                                           F16_src_fmt;
        B1_F16_2_operands: cross FP_src2_ops,   F16_src1_basictypes, F16_src2_basictypes,                      F16_src_fmt;
        B1_F16_3_operands: cross FP_src3_ops,   F16_src1_basictypes, F16_src2_basictypes, F16_src3_basictypes, F16_src_fmt;
        B1_F16_result:     cross FP_result_ops, F16_result_basictypes,                                         F16_result_fmt;
    `endif // COVER_F16

    `ifdef COVER_BF16
        B1_BF16_1_operands: cross FP_src1_ops,   BF16_src1_basictypes,                                             BF16_src_fmt;
        B1_BF16_2_operands: cross FP_src2_ops,   BF16_src1_basictypes, BF16_src2_basictypes,                       BF16_src_fmt;
        B1_BF16_3_operands: cross FP_src3_ops,   BF16_src1_basictypes, BF16_src2_basictypes, BF16_src3_basictypes, BF16_src_fmt;
        B1_BF16_result:     cross FP_result_ops, BF16_result_basictypes,                                           BF16_result_fmt;
    `endif // COVER_BF16

    
    `ifdef COVER_F128
        B1_F128_1_operands: cross FP_src1_ops,   F128_src1_basictypes,                                             F128_src_fmt;
        B1_F128_2_operands: cross FP_src2_ops,   F128_src1_basictypes, F128_src2_basictypes,                       F128_src_fmt;
        B1_F128_3_operands: cross FP_src3_ops,   F128_src1_basictypes, F128_src2_basictypes, F128_src3_basictypes, F128_src_fmt;
        B1_F128_result:     cross FP_result_ops, F128_result_basictypes,                                           F128_result_fmt;
    `endif // COVER_F128


endgroup