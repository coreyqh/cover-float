`include "coverfloat_pkg.sv"
`include "coverfloat_interface.sv"
`include "coverfloat_coverage.sv"

module coverfloat (); import coverfloat_pkg::*; // TODO: maybe rename...

    logic clk = 0;
    logic [31:0] vectornum;
    logic [`COVER_VECTOR_WIDTH - 1:0] covervectors;
    logic [2:0] discard; // bits we dont car about (upper 3 bits of sign nibble in vectors)

    coverfloat_coverage coverage_inst;
    coverfloat_interface CFI();

    initial begin

        int fd;
        fd = $fopen("../tests/covervectors/B1_cv.txt", "r");

        vectornum = 0;
        
        // CFI           = new();
        coverage_inst = new(CFI);

        while ($fscanf(fd, "%h", covervectors) == 1) begin
            @(posedge clk);
        end
        @(negedge clk);
        $fclose(fd);
        $stop;

    end

    initial begin
        clk = 0; forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        {CFI.op, CFI.rm, CFI.a, CFI.b, CFI.c, CFI.operandFmt, CFI.result, 
         CFI.resultFmt, CFI.exceptionBits, discard[2:0], CFI.intermS, CFI.intermX, CFI.intermM} = covervectors;
    end

    always @(negedge clk) begin
        // collect coverage 
        coverage_inst.sample();

        // $display("%h (%h) %h = %h_%h_%h", CFI.a, CFI.op, CFI.b, CFI.intermS, CFI.intermX, CFI.intermM[191:30]);

        vectornum = vectornum + 1;

    end

endmodule