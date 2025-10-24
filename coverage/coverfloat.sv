module coverfloat (coverfloat_interface CFI); import coverfloat_pkg::*;

    logic clk;
    logic [31:0] vectornum;
    logic [ghghgh:0] covervectors [10000:0];

    coverfloat_coverage  coverage_inst;

    initial begin

        $readmemb("covervectors.txt", covervectors);

        vectornum = 0;
        
        coverage_inst = new(CFI);

    end

    initial begin
        clk = 0; forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        {CFI.op, CFI.rm, CFI.a, CFI.b, CFI.c, CFI.aFmt, CFI.bFmt, CFI.cFmt, CFI.result, 
         CFI.resultFmt, CFI.intermS, CFI.intermX, CFI.intermM, CFI.exceptionBits}       = covervectors[vectornum];
    end

    always @(negedge clk) begin
        // collect coverage 
        coverage_inst.sample();

        vectornum = vectornum + 1;

    end

endmodule