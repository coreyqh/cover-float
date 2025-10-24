class coverfloat_coverage; import coverfloat_pkg::*;

    virtual coverfloat_interface CFI;
    B1_cg B1;

    // constructor (initializes covergroups)
    function new (virtual coverfloat_interface CFI);
        this.CFI = CFI;

        // initialize covergroups
        B1 = new(CFI);
        // ...

    endfunction

    
    function void sample();
        
        // Call sample functions (probably `include 'd)
        B1.sample();
        // ...

    endfunction

endclass