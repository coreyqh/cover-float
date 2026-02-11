def generate_test_vector(op: str, in1: int, in2: int, in3: int, fmt1: str, fmt2: str, rnd_mode: str = "00"):
    zero_padding = '0' * 32
    return f"{op}_{rnd_mode}_{in1:032x}_{in2:032x}_{in3:032x}_{fmt1}_{zero_padding}_{fmt2}_00\n"