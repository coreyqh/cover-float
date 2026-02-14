"""
Parse test vectors into human-readable floating-point format.

Converts hex-encoded test vectors into readable format like:
b32+ =0 -1.016A3DP101 +1.7CEE72P95 -> -1.7AED06P100 x

Currently only supports 
- Rounding mode: Round to Nearest Even
- Operations: add, sub, mul, div, fmadd, fmsub, fnmadd, fnmsub, sqrt, rem
- Flags: 'x' if a flag is raised and '' if none
"""

import sys
from pathlib import Path

# Format specifications
FMT_SPECS = {
    "00": {"name": "f16", "exp_bits": 5, "man_bits": 10, "bias": 15, "total_bits": 16},
    "01": {"name": "f32", "exp_bits": 8, "man_bits": 23, "bias": 127, "total_bits": 32},
    "02": {"name": "f64", "exp_bits": 11, "man_bits": 52, "bias": 1023, "total_bits": 64},
    "03": {"name": "f128", "exp_bits": 15, "man_bits": 112, "bias": 16383, "total_bits": 128},
    "04": {"name": "bf16", "exp_bits": 8, "man_bits": 7, "bias": 127, "total_bits": 16},
}

OP_NAMES = {
    "00000010": "add",
    "00000020": "sub",
    "00000030": "mul",
    "00000040": "div",
    "00000051": "fmadd",
    "00000052": "fmsub",
    "00000053": "fnmadd",
    "00000054": "fnmsub",
    "00000060": "sqrt",
    "00000070": "rem"
}

ROUND_NAMES = {
    "00": "=0",  # RNE (Round to Nearest Even)
}


def hex_to_binary(hex_str, bits):
    """Convert hex string to binary string of specified length."""
    return bin(int(hex_str, 16))[2:].zfill(bits)


def parse_fp_value(hex_val, fmt_code):
    """
    Parse a hex value into components based on format.
    Returns: (sign, exponent, mantissa, is_zero, is_inf, is_nan, is_subnormal)
    """
    spec = FMT_SPECS.get(fmt_code)
    if not spec:
        return None
    
    # Get the full value with proper bit width
    total_bits = spec["total_bits"]
    val = int(hex_val, 16)
    
    # Extract components
    sign = (val >> (total_bits - 1)) & 1
    exp_bits = spec["exp_bits"]
    man_bits = spec["man_bits"]
    
    biased_exp = (val >> man_bits) & ((1 << exp_bits) - 1)
    mantissa = val & ((1 << man_bits) - 1)
    
    # Determine special cases
    is_zero = (biased_exp == 0 and mantissa == 0)
    is_inf = (biased_exp == ((1 << exp_bits) - 1) and mantissa == 0)
    is_nan = (biased_exp == ((1 << exp_bits) - 1) and mantissa != 0)
    is_subnormal = (biased_exp == 0 and mantissa != 0)
    
    # Compute actual exponent
    if is_zero or is_subnormal:
        actual_exp = 1 - spec["bias"]
    else:
        actual_exp = biased_exp - spec["bias"]
    
    return {
        "sign": sign,
        "exp": actual_exp,
        "mantissa": mantissa,
        "man_bits": man_bits,
        "is_zero": is_zero,
        "is_inf": is_inf,
        "is_nan": is_nan,
        "is_subnormal": is_subnormal,
    }


def format_mantissa(mantissa, man_bits):
    """Format mantissa as hex with leading 1."""
    if mantissa == 0:
        return "0.000000"
    
    # Convert to hex string with leading 1 (implicit bit)
    hex_str = f"{mantissa:X}"
    
    # Pad to appropriate length
    hex_digits = (man_bits + 3) // 4  # Round up to nearest hex digit
    hex_str = hex_str.zfill(hex_digits)
    
    # Insert decimal point after first digit
    if len(hex_str) > 1:
        return "1." + hex_str[:6].ljust(6, '0')
    else:
        return "1.000000"


def fp_to_string(parsed_fp, fmt_code):
    """Convert parsed FP to human-readable string."""
    if parsed_fp["is_nan"]:
        return "NaN"
    if parsed_fp["is_inf"]:
        sign_char = "-" if parsed_fp["sign"] else "+"
        return f"{sign_char}Inf"
    if parsed_fp["is_zero"]:
        sign_char = "-" if parsed_fp["sign"] else "+"
        return f"{sign_char}0.000000P0"
    
    sign_char = "-" if parsed_fp["sign"] else "+"
    mantissa_str = format_mantissa(parsed_fp["mantissa"], parsed_fp["man_bits"])
    exp_str = f"P{parsed_fp['exp']}"
    
    return f"{sign_char}{mantissa_str}{exp_str}"


def parse_test_vector(line):
    """
    Parse a single test vector line.
    Format: OP_RM_A_B_C_OPFMT_RESULT_RESFMT_FLAGS
    Where A, B, C, RESULT are hex values of variable width based on format
    """
    line = line.strip()
    if not line or line.startswith("//"):
        return None
    
    parts = line.split("_")
    if len(parts) < 9:
        return None
    
    try:
        op_code = parts[0]
        rnd_code = parts[1]
        a_val = parts[2]
        b_val = parts[3]
        c_val = parts[4]  # Unused for ADD/SUB
        op_fmt = parts[5]  # Format for operands
        result_val = parts[6]
        result_fmt = parts[7]
        flags = parts[8] if len(parts) > 8 else "00"
        
        op_name = OP_NAMES.get(op_code, "?")
        rnd_name = ROUND_NAMES.get(rnd_code, "?")
        
        # Determine the number of hex characters needed for each format
        op_spec = FMT_SPECS.get(op_fmt)
        res_spec = FMT_SPECS.get(result_fmt)
        if not op_spec or not res_spec:
            return None
        
        op_hex_chars = op_spec["total_bits"] // 4
        res_hex_chars = res_spec["total_bits"] // 4
        
        # Extract operands using the correct bit width
        a_val_formatted = a_val[-op_hex_chars:] if len(a_val) >= op_hex_chars else a_val.zfill(op_hex_chars)
        b_val_formatted = b_val[-op_hex_chars:] if len(b_val) >= op_hex_chars else b_val.zfill(op_hex_chars)
        result_val_formatted = result_val[-res_hex_chars:] if len(result_val) >= res_hex_chars else result_val.zfill(res_hex_chars)
        
        a_parsed = parse_fp_value(a_val_formatted, op_fmt)
        b_parsed = parse_fp_value(b_val_formatted, op_fmt)
        result_parsed = parse_fp_value(result_val_formatted, result_fmt)
        
        if not a_parsed or not b_parsed or not result_parsed:
            return None
        
        a_str = fp_to_string(a_parsed, op_fmt)
        b_str = fp_to_string(b_parsed, op_fmt)
        result_str = fp_to_string(result_parsed, result_fmt)
        
        # Determine format code for output
        fmt_name = op_spec["name"]

        options = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "fmadd": "*+",
            "fmsub": "*-",
            "fnmadd": "-*+",
            "fnmsub": "-*-",
            "sqrt": "v-",
            "rem": "mod"
        }
        op_sym = options.get(op_name)
        
        # Determine flags output
        flags_str = "x" if flags != "00" else ""
        
        return {
            "format": f"{fmt_name}{op_sym}",
            "round": rnd_name,
            "op_a": a_str,
            "op_b": b_str,
            "result": result_str,
            "flags": flags_str,
            "full_line": line,
        }
    except (ValueError, IndexError):
        return None


def format_output(parsed):
    """Format parsed test vector to output string."""
    flags = f" {parsed['flags']}" if parsed['flags'] else ""
    return f"{parsed['format']} {parsed['round']} {parsed['op_a']} {parsed['op_b']} -> {parsed['result']}{flags}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_testvectors.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    results = []
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            parsed = parse_test_vector(line)
            if parsed:
                output_str = format_output(parsed)
                results.append(output_str)
    
    if output_file:
        with open(output_file, "w") as f:
            for result in results:
                f.write(result + "\n")
        print(f"Parsed {len(results)} test vectors to {output_file}")
    else:
        for result in results[:10]:  # Print first 10
            print(result)
        if len(results) > 10:
            print(f"... ({len(results) - 10} more vectors)")
        print(f"\nTotal: {len(results)} test vectors")


if __name__ == "__main__":
    main()
