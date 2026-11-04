library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

package ds_test_vectors_pkg is

    constant NUM_CASES   : natural := 9;
    constant MAX_SAMPLES : natural := 16;

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
        0 => 1,
        1 => 1,
        2 => 1,
        3 => 2,
        4 => 2,
        5 => 4,
        6 => 4,
        7 => 5,
        8 => 16
    );

    constant SAMPLES : sample_matrix_t := (
        0 => (0 => x"0000", others => x"0000"),
        1 => (0 => x"007B", others => x"0000"),
        2 => (0 => x"FF85", others => x"0000"),
        3 => (0 => x"03E8", 1 => x"FC18", others => x"0000"),
        4 => (0 => x"7FFF", 1 => x"8000", others => x"0000"),
        5 => (0 => x"FFFF", 1 => x"FFFE", 2 => x"FFFD", 3 => x"FFFC", others => x"0000"),
        6 => (0 => x"7530", 1 => x"8AD0", 2 => x"4E20", 3 => x"B1E0", others => x"0000"),
        7 => (0 => x"0007", 1 => x"FFFC", 2 => x"0013", 3 => x"FFEA", 4 => x"0003", others => x"0000"),
        8 => (0 => x"B617", 1 => x"B56B", 2 => x"9C57", 3 => x"0BCA", 4 => x"FC6D", 5 => x"F51B", 6 => x"629D", 7 => x"DA4E", 8 => x"9986", 9 => x"DBED", 10 => x"FB7A", 11 => x"B226", 12 => x"8A7A", 13 => x"C495", 14 => x"181B", 15 => x"96A2")
    );

    constant EXPECTED_D : result_array_t := (
        0 => x"0000000000000000",
        1 => x"000000000000007B",
        2 => x"FFFFFFFFFFFFFF85",
        3 => x"0000000000000000",
        4 => x"FFFFFFFFFFFFFFFF",
        5 => x"FFFFFFFFFFFFFFF6",
        6 => x"0000000000000000",
        7 => x"0000000000000003",
        8 => x"FFFFFFFFFFFD62F5"
    );

    constant EXPECTED_S : result_array_t := (
        0 => x"0000000000000000",
        1 => x"0000000000003B19",
        2 => x"0000000000003B19",
        3 => x"00000000001E8480",
        4 => x"000000007FFF0001",
        5 => x"000000000000001E",
        6 => x"000000009AF8DA00",
        7 => x"0000000000000397",
        8 => x"0000000135B2FE9D"
    );

end package ds_test_vectors_pkg;
