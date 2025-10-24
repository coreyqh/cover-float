covergroup B1_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    op: coverpoint CFI.op {
        `include "op_bins.svh"
    }

    F32_src1_fmt: coverpoint CFI.aFmt == FMT_SINGLE {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F32_src2_fmt: coverpoint CFI.bFmt == FMT_SINGLE {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F32_src3_fmt: coverpoint CFI.cFmt == FMT_SINGLE {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F32_result_fmt: coverpoint CFI.resultFmt == FMT_SINGL E {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F32_src1_basictypes: coverpoint CFI.a[31:0]; {
        type_option.weight = 0;
        bins pos0             = {32'h00000000};
        bins neg0             = {32'h80000000};
        bins pos1             = {32'h3f800000};
        bins neg1             = {32'hbf800000};
        bins pos1p5           = {32'h3fc00000};
        bins neg1p5           = {32'hbfc00000};
        bins pos2             = {32'h40000000};
        bins neg2             = {32'hc0000000};
        bins posminnorm       = {32'h00800000};
        bins mnegminnorm      = {32'h80800000};
        bins posmaxnorm       = {32'h7f7fffff};
        bins negmaxnorm       = {32'hff7fffff};
        bins posnorm          = {[32'h00800000:32'h7f7fffff]};
        bins negnorm          = {[32'h80800000:32'hff7fffff]};
        bins posmax_subnorm   = {32'h007fffff};
        bins negmax_subnorm   = {32'h807fffff};
        bins posmid_subnorm   = {32'h00400000};
        bins negmid_subnorm   = {32'h80400000};
        bins posmin_subnorm   = {32'h00000001};
        bins negmin_subnorm   = {32'h80000001};
        bins pos_subnorm      = {[32'h00000001:32'h007fffff]};
        bins neg_subnorm      = {[32'h80000001:32'h807fffff]};
        bins posinfinity      = {32'h7f800000};
        bins neginfinity      = {32'hff800000};
        bins posQNaN          = {[32'h7fc00000:32'h7fffffff]};
        bins posSNaN          = {[32'h7f800001:32'h7fbfffff]};
        bins negQNaN          = {[32'hffc00000:32'hffffffff]};
        bins negSNaN          = {[32'hff800001:32'hffbfffff]};
    }

    F32_src2_basictypes: coverpoint CFI.b[31:0]; {
        type_option.weight = 0;
        bins pos0             = {32'h00000000};
        bins neg0             = {32'h80000000};
        bins pos1             = {32'h3f800000};
        bins neg1             = {32'hbf800000};
        bins pos1p5           = {32'h3fc00000};
        bins neg1p5           = {32'hbfc00000};
        bins pos2             = {32'h40000000};
        bins neg2             = {32'hc0000000};
        bins posminnorm       = {32'h00800000};
        bins mnegminnorm      = {32'h80800000};
        bins posmaxnorm       = {32'h7f7fffff};
        bins negmaxnorm       = {32'hff7fffff};
        bins posnorm          = {[32'h00800000:32'h7f7fffff]};
        bins negnorm          = {[32'h80800000:32'hff7fffff]};
        bins posmax_subnorm   = {32'h007fffff};
        bins negmax_subnorm   = {32'h807fffff};
        bins posmid_subnorm   = {32'h00400000};
        bins negmid_subnorm   = {32'h80400000};
        bins posmin_subnorm   = {32'h00000001};
        bins negmin_subnorm   = {32'h80000001};
        bins pos_subnorm      = {[32'h00000001:32'h007fffff]};
        bins neg_subnorm      = {[32'h80000001:32'h807fffff]};
        bins posinfinity      = {32'h7f800000};
        bins neginfinity      = {32'hff800000};
        bins posQNaN          = {[32'h7fc00000:32'h7fffffff]};
        bins posSNaN          = {[32'h7f800001:32'h7fbfffff]};
        bins negQNaN          = {[32'hffc00000:32'hffffffff]};
        bins negSNaN          = {[32'hff800001:32'hffbfffff]};
    }

    F32_src3_basictypes: coverpoint CFI.c[31:0]; {
        type_option.weight = 0;
        bins pos0             = {32'h00000000};
        bins neg0             = {32'h80000000};
        bins pos1             = {32'h3f800000};
        bins neg1             = {32'hbf800000};
        bins pos1p5           = {32'h3fc00000};
        bins neg1p5           = {32'hbfc00000};
        bins pos2             = {32'h40000000};
        bins neg2             = {32'hc0000000};
        bins posminnorm       = {32'h00800000};
        bins mnegminnorm      = {32'h80800000};
        bins posmaxnorm       = {32'h7f7fffff};
        bins negmaxnorm       = {32'hff7fffff};
        bins posnorm          = {[32'h00800000:32'h7f7fffff]};
        bins negnorm          = {[32'h80800000:32'hff7fffff]};
        bins posmax_subnorm   = {32'h007fffff};
        bins negmax_subnorm   = {32'h807fffff};
        bins posmid_subnorm   = {32'h00400000};
        bins negmid_subnorm   = {32'h80400000};
        bins posmin_subnorm   = {32'h00000001};
        bins negmin_subnorm   = {32'h80000001};
        bins pos_subnorm      = {[32'h00000001:32'h007fffff]};
        bins neg_subnorm      = {[32'h80000001:32'h807fffff]};
        bins posinfinity      = {32'h7f800000};
        bins neginfinity      = {32'hff800000};
        bins posQNaN          = {[32'h7fc00000:32'h7fffffff]};
        bins posSNaN          = {[32'h7f800001:32'h7fbfffff]};
        bins negQNaN          = {[32'hffc00000:32'hffffffff]};
        bins negSNaN          = {[32'hff800001:32'hffbfffff]};
    }

    F32_result_basictypes: coverpoint CFI.result[31:0]; {
        type_option.weight = 0;
        bins pos0             = {32'h00000000};
        bins neg0             = {32'h80000000};
        bins pos1             = {32'h3f800000};
        bins neg1             = {32'hbf800000};
        bins pos1p5           = {32'h3fc00000};
        bins neg1p5           = {32'hbfc00000};
        bins pos2             = {32'h40000000};
        bins neg2             = {32'hc0000000};
        bins posminnorm       = {32'h00800000};
        bins mnegminnorm      = {32'h80800000};
        bins posmaxnorm       = {32'h7f7fffff};
        bins negmaxnorm       = {32'hff7fffff};
        bins posnorm          = {[32'h00800000:32'h7f7fffff]};
        bins negnorm          = {[32'h80800000:32'hff7fffff]};
        bins posmax_subnorm   = {32'h007fffff};
        bins negmax_subnorm   = {32'h807fffff};
        bins posmid_subnorm   = {32'h00400000};
        bins negmid_subnorm   = {32'h80400000};
        bins posmin_subnorm   = {32'h00000001};
        bins negmin_subnorm   = {32'h80000001};
        bins pos_subnorm      = {[32'h00000001:32'h007fffff]};
        bins neg_subnorm      = {[32'h80000001:32'h807fffff]};
        bins posinfinity      = {32'h7f800000};
        bins neginfinity      = {32'hff800000};
        bins posQNaN          = {[32'h7fc00000:32'h7fffffff]};
        bins posSNaN          = {[32'h7f800001:32'h7fbfffff]};
        bins negQNaN          = {[32'hffc00000:32'hffffffff]};
        bins negSNaN          = {[32'hff800001:32'hffbfffff]};
    }

    // main coverpoints

    `ifdef COVER_F32
        B1_F32_1_operands: cross op_1,   F32_src1_basictypes, F32_src1_fmt;
        B1_F32_2_operands: cross op_2,   F32_src1_basictypes, F32_src2_basictypes, F32_src1_fmt, F32_src2_fmt;
        B1_F32_3_operands: cross op_3,   F32_src1_basictypes, F32_src2_basictypes, F32_src3_basictypes, F32_src1_fmt, F32_src2_fmt, F32_src3_fmt;
        B1_F32_result:     cross op_all, F32_result_basictypes, F32_result_fmt;
    `endif // COVER_F32

    `ifdef COVER_F32
    `ifdef COVER_F64
        B1_F32_F64_operand: cross 
        B1_F64_F32_operand: cross
        B1_F32_F64_result:  cross 
        B1_F64_F32_result:  cross
    `endif 
    `endif // COVER_F32 && COVER_F64
endgroup