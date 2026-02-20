"""
Parse test vectors into human-readable floating-point format.

Converts hex-encoded test vectors into readable format like:
b32+ =0 -1.016A3DP101 +1.7CEE72P95 -> -1.7AED06P100 x

Currently supports 
- Rounding mode: Round to Nearest Even
- Operations: add, sub, mul, div, fmadd, fmsub, fnmadd, fnmsub, sqrt, rem, cfi, cff, cif, class
- Flags: 'x' if a flag is raised and '' if none
"""

import sys
from pathlib import Path

FMT_SPECS = {
    "00": {"name": "f16", "type": "float", "exp_bits": 5, "man_bits": 10, "bias": 15, "total_bits": 16},
    "01": {"name": "f32", "type": "float", "exp_bits": 8, "man_bits": 23, "bias": 127, "total_bits": 32},
    "02": {"name": "f64", "type": "float", "exp_bits": 11, "man_bits": 52, "bias": 1023, "total_bits": 64},
    "03": {"name": "f128", "type": "float", "exp_bits": 15, "man_bits": 112, "bias": 16383, "total_bits": 128},
    "04": {"name": "bf16", "type": "float", "exp_bits": 8, "man_bits": 7, "bias": 127, "total_bits": 16},
    "81": {"name": "int",  "type": "int",  "signed": True,  "total_bits": 32},
    "c1": {"name": "uint", "type": "int",  "signed": False, "total_bits": 32},
    "82": {"name": "long", "type": "int",  "signed": True,  "total_bits": 64},
    "c2": {"name": "ul",   "type": "int",  "signed": False, "total_bits": 64},
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
    "00000070": "rem",
    "00000080": "cfi",
    "00000090": "cff",
    "000000A0": "cif",
    "000000D0": "class"
}

ROUND_NAMES = {
    "00": "=0",  # RNE (Round to Nearest Even)
}


def hex_to_binary(hex_str, bits):
    """Convert hex string to binary string of specified length."""
    return bin(int(hex_str, 16))[2:].zfill(bits)


def parse_int_value(hex_val, spec):
    """Interpret a hex string according to an integer spec."""
    bits = spec["total_bits"]
    unsigned = int(hex_val, 16)
    if spec.get("signed"):
        sign_bit = 1 << (bits - 1)
        value = unsigned - (1 << bits) if (unsigned & sign_bit) else unsigned
    else:
        value = unsigned
    return {"value": value, "signed": spec.get("signed", False), "raw": hex_val}


def parse_fp_value(hex_val, fmt_code):
    """
    Parse a hex value into components based on format.
    Returns: (sign, exponent, mantissa, is_zero, is_inf, is_nan, is_subnormal)
    """
    spec = FMT_SPECS.get(fmt_code)
    if not spec:
        return None
    
    total_bits = spec["total_bits"]
    val = int(hex_val, 16)
    
    sign = (val >> (total_bits - 1)) & 1
    exp_bits = spec["exp_bits"]
    man_bits = spec["man_bits"]
    
    biased_exp = (val >> man_bits) & ((1 << exp_bits) - 1)
    mantissa = val & ((1 << man_bits) - 1)
    
    is_zero = (biased_exp == 0 and mantissa == 0)
    is_inf = (biased_exp == ((1 << exp_bits) - 1) and mantissa == 0)
    is_nan = (biased_exp == ((1 << exp_bits) - 1) and mantissa != 0)
    is_subnormal = (biased_exp == 0 and mantissa != 0)
    
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

def format_mantissa(parsed):
    """Format mantissa as hex with leading 1 for normal, 0 for subnormal."""
    mantissa = parsed["mantissa"]
    man_bits = parsed["man_bits"]
    
    # Determine the hidden bit
    lead_bit = "0" if parsed["is_subnormal"] else "1"
    
    if mantissa == 0:
        return f"{lead_bit}.0"

    hex_str = f"{mantissa:X}"
    hex_digits = (man_bits + 3) // 4
    hex_str = hex_str.zfill(hex_digits)

    return f"{lead_bit}.{hex_str}"


def decode_class_mask(val):
    """Decodes fclass bitmask."""
    masks = {
        0: "NegInf", 1: "NegNormal", 2: "NegSubnormal", 3: "NegZero",
        4: "PosZero", 5: "PosSubnormal", 6: "PosNormal", 7: "PosInf",
        8: "sNaN", 9: "qNaN"
    }
    active = [name for bit, name in masks.items() if (val >> bit) & 1]
    return "|".join(active) if active else hex(val)


def value_to_string(parsed, fmt_code, is_class = False):
    """Format a parsed value (float or int) into a string.

    Floats get the usual ±mantissaPexponent representation; integers are
    shown in decimal (preserving sign) with a leading `0x` prefix for clarity.
    """
    spec = FMT_SPECS.get(fmt_code, {})

    if is_class:
        return decode_class_mask(parsed["value"])

    if spec.get("type") == "int":
        val = parsed["value"]
        if not parsed.get("signed"):
            return hex(parsed["value"])
        return str(val)

    if parsed["is_nan"]:
        return "NaN"
    if parsed["is_inf"]:
        sign_char = "-" if parsed["sign"] else "+"
        return f"{sign_char}Inf"
    if parsed["is_zero"]:
        sign_char = "-" if parsed["sign"] else "+"
        return f"{sign_char}0.0P0"

    sign_char = "-" if parsed["sign"] else "+"
    mantissa_str = format_mantissa(parsed)
    exp_str = f"P{parsed['exp']}"

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

    op_code = parts[0]
    rnd_code = parts[1]
    a_val = parts[2]
    b_val = parts[3]
    c_val = parts[4]
    op_fmt = parts[5]
    result_val = parts[6]
    result_fmt = parts[7]
    flags = parts[8] if len(parts) > 8 else "00"
    
    # Define operation categories
    one_op_names = ("sqrt", "cfi", "cff", "cif", "class")
    three_op_names = ("fmadd", "fmsub", "fnmadd", "fnmsub")

    op_name = OP_NAMES.get(op_code.upper(), "?")
    rnd_name = ROUND_NAMES.get(rnd_code, "?")

    op_spec = FMT_SPECS.get(op_fmt)
    res_spec = FMT_SPECS.get(result_fmt)
    if not op_spec or not res_spec:
        print(f"This format is current not supported by the parsing script: op_fmt={op_fmt!r} res_fmt={result_fmt!r} in line {line!r}")
        return None

    op_hex_chars = op_spec["total_bits"] // 4
    res_hex_chars = res_spec["total_bits"] // 4

    def fixwidth(val, width):
        return val[-width:] if len(val) >= width else val.zfill(width)

    a_val_formatted = fixwidth(a_val, op_hex_chars)
    b_val_formatted = fixwidth(b_val, op_hex_chars)
    c_val_formatted = fixwidth(c_val, op_hex_chars)
    result_val_formatted = fixwidth(result_val, res_hex_chars)

    try:
        if op_spec.get("type") == "float":
            a_parsed = parse_fp_value(a_val_formatted, op_fmt)
        else:
            a_parsed = parse_int_value(a_val_formatted, op_spec)

        # Only parse B if it's not a 1-operand operation
        b_parsed = None
        if op_name not in one_op_names:
            if op_spec.get("type") == "float":
                b_parsed = parse_fp_value(b_val_formatted, op_fmt)
            else:
                b_parsed = parse_int_value(b_val_formatted, op_spec)

        # Parse C only for 3-operand operations
        c_parsed = None
        if op_name in three_op_names:
            c_parsed = parse_fp_value(c_val_formatted, op_fmt)

        if op_name == "class":
            result_parsed = parse_int_value(result_val_formatted, {"total_bits": 32, "signed": False})
        elif res_spec and res_spec.get("type") == "float":
            result_parsed = parse_fp_value(result_val_formatted, result_fmt)
        elif res_spec:
            result_parsed = parse_int_value(result_val_formatted, res_spec)
        else:
            result_parsed = None
    except Exception as err:
        print(f"warning: failed to parse line {line!r}: {err}")
        return None

    a_str = value_to_string(a_parsed, op_fmt)
    b_str = value_to_string(b_parsed, op_fmt) if b_parsed else None
    c_str = value_to_string(c_parsed, op_fmt) if c_parsed else None
    result_str = value_to_string(result_parsed, result_fmt, is_class = (op_name) == "class") if res_spec else result_val_formatted

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
        "rem": "rem",
        "cfi": "cfi",
        "cff": "cff",
        "cif": "cif",
        "class": "cls"
    }
    op_sym = options.get(op_name)
    flags_str = "x" if flags != "00" else ""

    result = {
        "format": f"{fmt_name}{op_sym}",
        "round": rnd_name,
        "op_a": a_str,
        "op_b": b_str,
        "result": result_str,
        "flags": flags_str,
        "full_line": line,
        "res_fmt_name": res_spec["name"] if res_spec else None,
    }
    if c_str:
        result["op_c"] = c_str
    return result

def format_output(parsed):
    """Format parsed test vector to output string based on operand count."""
    flags = f" {parsed['flags']}" if parsed['flags'] else ""
    op_name = parsed['format'] 

    if "v-" in op_name or any(x in op_name for x in ["cfi", "cff", "cif", "cls"]):
        base = f"{parsed['format']} {parsed['round']} {parsed['op_a']}"
    elif "op_c" in parsed:
        base = f"{parsed['format']} {parsed['round']} {parsed['op_a']} {parsed['op_b']} {parsed['op_c']}"
    else:
        base = f"{parsed['format']} {parsed['round']} {parsed['op_a']} {parsed['op_b']}"
    
    base += f" -> {parsed['result']}"
    if parsed.get('res_fmt_name'):
        base += f" ({parsed['res_fmt_name']})"
    return base + flags


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
        for result in results[:10]:
            print(result)
        if len(results) > 10:
            print(f"... ({len(results) - 10} more vectors)")
        print(f"\nTotal: {len(results)} test vectors")


if __name__ == "__main__":
    main()
