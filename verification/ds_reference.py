from pathlib import Path
import random

# run: python verification\ds_reference.py

# This script takes a set of PCM samples, computes expected D and S
# then writes samples and expected results to a VHDL package 
# for automated verification of ds_accum.vhd by the Vivado testbench

# OUTPUT = "verification/ds_test_vectors_pkg.vhd"
# saves generated VHDL next to this script regardless of current working directory
OUTPUT = Path(__file__).with_name("ds_test_vectors_pkg.vhd")

# Compute D and S. Finally return D and S
# Independent integer reference for the hardware D/S accumulator
def ds_reference(samples):
    d = 0
    s = 0

    # Check each input sample is within signed 16-bit PCM range (-32768 to 32767)
    # If x out of range, raise ValueError
    for x in samples:
        if not -32768 <= x <= 32767:
            raise ValueError(f"{x} is outside signed 16-bit range")
        d += x
        s += x * x
    return d, s

def to_hex(value, bits):
    # Convert Python integers to two's-complement hex representation
    # VHDL Pkg: std_logic_vector(63 downto 0) 64-bit vector
    # e.g. '-1' -> x"FFFFFFFF FFFFFFFF"
    if value < 0:
        value = (1 << bits) + value
    # One hexadecimal digit: 4 bits
    width = bits // 4
    return f"{value:0{width}X}"

# Test cases chosen independently from RTL author's testbench
cases = {
    "single_zero": [0],
    "single_positive": [123],
    "single_negative": [-123],
    "positive_negative_cancel": [1000, -1000], # D = 0, S > 0
    "pcm16_limits": [32767, -32768], # boundary case
    "negative_sequence": [-1, -2, -3, -4],
    "alternating_large": [30000, -30000, 20000, -20000],
    "mixed_values": [7, -4, 19, -22, 3],
}

# Random number generator: Reproducible random case (same seed 3601)
rng = random.Random(3601)
random_samples = []

for i in range(16):
    # Repeat 16 times: select an integer from valid PCM16 range (-32768 to 32767)
    # Append each to random_samples
    random_samples.append(
        rng.randint(-32768, 32767)
    )
# 9th test case: pseudo-random 16 samples
cases["random_16"] = random_samples

# Find max num of samples across all test cases (e.g. 16)
# VHDL array 'sample_row_t': uniform lengths; zero-pad shorter cases
max_samples = max(len(samples) for samples in cases.values())

length_lines = []
sample_lines = []
expected_d_lines = []
expected_s_lines = []

# cases.items(): extract name and samples from each dictionary entry
# enumerate(...): assign index to each case
for index, (name, samples) in enumerate(cases.items()):
    d, s = ds_reference(samples) # Compute expected D and S

    length_lines.append(
        # eg. 3 => 2 (case 3: 2 valid samples)
        # f-string. Record actual sample count per case
        # TB ignore zero-padding, processing only original samples
        f"        {index} => {len(samples)}"
    )

    entries = []

    for i, value in enumerate(samples):
        # enumerate(samples) iterate over samples: each index i and value
        entries.append(
            f'{i} => x"{to_hex(value, 16)}"' # PCM sample: 16 bits
        )

    if len(samples) < max_samples:
        # eg.
        # 3 => (
        #   0 => x"03E8", 1 => x"FC18", others => x"0000"
        # )
        entries.append('others => x"0000"')

    sample_lines.append(
        f"        {index} => ({', '.join(entries)})"
    )

    expected_d_lines.append(
        f'        {index} => x"{to_hex(d, 64)}"'
    )

    expected_s_lines.append(
        f'        {index} => x"{to_hex(s, 64)}"'
    )

    # For terminal output
    print(
        f"{index}: {name}\n"
        f"   samples = {samples}\n"
        f"   D = {d}\n"
        f"   S = {s}"
    )

# multi-line f-string
# Triple quotes (""") allow strings to span multiple lines
vhdl = f"""library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

package ds_test_vectors_pkg is

    constant NUM_CASES   : natural := {len(cases)};
    constant MAX_SAMPLES : natural := {max_samples};

    type sample_row_t is array (
        0 to MAX_SAMPLES - 1
    ) of std_logic_vector(15 downto 0);

    type sample_matrix_t is array (
        0 to NUM_CASES - 1
    ) of sample_row_t;

    type length_array_t is array (
        0 to NUM_CASES - 1
    ) of natural;

    type result_array_t is array (
        0 to NUM_CASES - 1
    ) of std_logic_vector(63 downto 0);

    constant CASE_LENGTHS : length_array_t := (
{",\n".join(length_lines)}
    );

    constant SAMPLES : sample_matrix_t := (
{",\n".join(sample_lines)}
    );

    constant EXPECTED_D : result_array_t := (
{",\n".join(expected_d_lines)}
    );

    constant EXPECTED_S : result_array_t := (
{",\n".join(expected_s_lines)}
    );

end package ds_test_vectors_pkg;
"""

OUTPUT.write_text(vhdl, encoding="utf-8")

print()
print(f"Generated: {OUTPUT}")