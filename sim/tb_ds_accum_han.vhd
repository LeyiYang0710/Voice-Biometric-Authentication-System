library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

use work.ds_test_vectors_pkg.all;

-- Simulation: run for 5 us to complete all verification cases

-- 1. Load test samples and expected D/S from Python-generated VHDL pkg
-- 2. Feed samples one by one into ds_accum
-- 3. Compare RTL computed D/S against expected values

entity tb_ds_accum_independent is
end tb_ds_accum_independent;

architecture Behavioral of tb_ds_accum_independent is

    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal clear : std_logic := '0';
    signal en : std_logic := '0'; -- Whether current x is a valid sample

    signal x : std_logic_vector(15 downto 0) := (others => '0');

    signal d : std_logic_vector(63 downto 0);
    signal s : std_logic_vector(63 downto 0);
    signal d_next : std_logic_vector(63 downto 0);
    signal s_next : std_logic_vector(63 downto 0);

begin
    clk <= not clk after 10 ns; -- Period = 20 ns
    -- Device Under Test: ds_accum
    dut : entity work.ds_accum
        port map (
            clk => clk,
            rst => rst,
            clear => clear,
            en => en,
            x => x,
            d => d,
            s => s,
            d_next => d_next,
            s_next => s_next
        );

    process
        -- First procedure: calling 'clear_accumulator' executes entire block below
        -- Assert clear; wait for next rising edge, deassert clear
        -- ds_accum checks whether clear = '1' only at each rising edge
        procedure clear_accumulator is
        begin
            -- clear = '1' does not reset D/S immediately
            -- Reset occurs only when DUT samples clear = '1' at a rising edge
            clear <= '1';
            wait until rising_edge(clk); -- DUT detects clear = '1' at the rising edge: resets D/S to 0
            -- After the rising edge, deasserts clear: clear complete
            clear <= '0';
            wait until rising_edge(clk);
        end procedure;

        procedure apply_sample(
            sample_value : std_logic_vector(15 downto 0)
        ) is
        begin
            x <= sample_value;
            en <= '1';

            wait until rising_edge(clk);

            en <= '0';
            x <= (others => '0');

            wait until rising_edge(clk);
        end procedure;

    begin

        -- Active low reset
        rst <= '0';
        wait for 100 ns;

        wait until rising_edge(clk);
        rst <= '1';

        wait until rising_edge(clk);

        -- Run every Python generated test case
        for case_index in 0 to NUM_CASES - 1 loop
            clear_accumulator;
            -- Feed samples from current case into DUT one by one
            -- SAMPLES: fixed maximum length
            -- CASE_LENGTHS: actual valid sample count per case, excluding zero-padding
            for sample_index in 0 to CASE_LENGTHS(case_index) - 1 loop
                apply_sample(
                    SAMPLES(case_index)(sample_index)
                );
            end loop;
            wait for 1 ns;

            -- RTL actual d/s vs Python-generated expected D/S
            assert d = EXPECTED_D(case_index)
                report
                    "D mismatch in case " & integer'image(case_index)
                severity error;
            assert s = EXPECTED_S(case_index)
                report
                    "S mismatch in case " & integer'image(case_index)
                severity error;
            report
                "Case " & integer'image(case_index) & " passed"
                severity note; -- Print All cases to Tcl Console

        end loop;

        report
            "All D/S independent verification cases passed"
            severity note;

        wait;

    end process;

end Behavioral;