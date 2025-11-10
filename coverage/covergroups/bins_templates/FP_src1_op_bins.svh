        // all operations where the first operand is FP

        // 1 fp input
        bins op_sqrt   = {OP_SQRT & ~32'hF};
        bins op_cfi    = {OP_CFI & ~32'hF};
        bins op_fcvtw  = {OP_FCVTW};
        bins op_fcvtwu = {OP_FCVTWU};
        bins op_fcvtl  = {OP_FCVTL};
        bins op_fcvtlu = {OP_FCVTLU};
        bins op_cff    = {OP_CFF & ~32'hF}; 
        bins op_class  = {OP_CLASS & ~32'hF};
        