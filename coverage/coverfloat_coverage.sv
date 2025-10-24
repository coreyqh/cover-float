class coverfloat_coverage; import coverfloat_pkg::*;

    virtual coverfloat_interface CFI;

    // constructor (initializes covergroups)
    function new (virtual coverfloat_interface CFI);
        this.CFI = CFI;

        // initialize covergroups

    endfunction

    
    function void sample();
        
        // Call sample functions (probably `include 'd)

    endfunction

endclass