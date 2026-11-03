library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

use work.ds_test_vectors_pkg.all;

-- Simulation: run for 10 us to complete all verification cases

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

        -- Second procedure: Feed sample into the RTL
        procedure apply_sample(
            sample_value : std_logic_vector(15 downto 0)
        ) is
        begin
            x <= sample_value;
            en <= '1';
            wait until rising_edge(clk);

            en <= '0'; -- Sample transfer complete, deassert en to mark x as invalid
            -- with en = '0' DUT normally ignores x regardless of its value
            x <= (others => '0'); -- Reset x for cleaner waveforms only
            -- Hold en = '0' for one clock cycle
            wait until rising_edge(clk); -- One idle clock cycle between consecutive valid samples
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

        -- Control Case 1: en = '0' must ignore a non-zero input
        -- Drive x = 12345 while en = '0'.
        clear_accumulator;

        x <= std_logic_vector(to_signed(12345, 16));
        en <= '0';

        wait until rising_edge(clk);
        wait for 1 ns;

        assert signed(d) = to_signed(0, 64)
            report "Control Case 1 failed: D changed when en=0"
            severity error;
        assert unsigned(s) = to_unsigned(0, 64)
            report "Control Case 1 failed: S changed when en=0"
            severity error;
        report "Control Case 1 passed: en=0 ignores input"
            severity note;

        -- Control Case 2: clear resets D/S and accumulation can restart afterwards
        clear_accumulator;

        apply_sample(std_logic_vector(to_signed(100, 16)));
        apply_sample(std_logic_vector(to_signed(-20, 16)));

        wait for 1 ns;

        assert signed(d) = to_signed(80, 64)
            report "Control Case 2 failed before clear: wrong D"
            severity error;
        assert unsigned(s) = to_unsigned(10400, 64)
            report "Control Case 2 failed before clear: wrong S"
            severity error;

        clear_accumulator;
        wait for 1 ns;

        assert signed(d) = to_signed(0, 64)
            report "Control Case 2 failed: clear did not reset D"
            severity error;
        assert unsigned(s) = to_unsigned(0, 64)
            report "Control Case 2 failed: clear did not reset S"
            severity error;

        -- Check accumulator still works after clear
        apply_sample(std_logic_vector(to_signed(7, 16)));
        wait for 1 ns;

        assert signed(d) = to_signed(7, 64)
            report "Control Case 2 failed: D did not restart after clear"
            severity error;
        assert unsigned(s) = to_unsigned(49, 64)
            report "Control Case 2 failed: S did not restart after clear"
            severity error;
        report "Control Case 2 passed: clear and restart"
            severity note;

        -- Control Case 3: active low reset clears D/S during operation
        clear_accumulator;

        apply_sample(std_logic_vector(to_signed(50, 16)));
        apply_sample(std_logic_vector(to_signed(25, 16)));

        -- rst is synchronous and active low
        rst <= '0';
        wait until rising_edge(clk);
        rst <= '1';
        wait for 1 ns;

        assert signed(d) = to_signed(0, 64)
            report "Control Case 3 failed: reset did not clear D"
            severity error;
        assert unsigned(s) = to_unsigned(0, 64)
            report "Control Case 3 failed: reset did not clear S"
            severity error;

        -- Check recovery after reset
        apply_sample(std_logic_vector(to_signed(-5, 16)));
        wait for 1 ns;

        assert signed(d) = to_signed(-5, 64)
            report "Control Case 3 failed: D did not recover after reset"
            severity error;
        assert unsigned(s) = to_unsigned(25, 64)
            report "Control Case 3 failed: S did not recover after reset"
            severity error;
        report "Control Case 3 passed: reset and recovery"
            severity note;

        -- Control Case 4: d_next/s_next predict the result of current x
        -- d_next/s_next are combinational, continuously calculated from current D/S and x
        -- d/s update only on a rising clock edge with en = '1'.
        -- Before the edge: expect d_next = -10, s_next = 100; d/s remain 0
        -- After the edge: expect d = -10, s = 100
        clear_accumulator;

        x  <= std_logic_vector(to_signed(-10, 16));
        en <= '1';

        -- d_next/s_next are combinational, so check before clock edge
        wait for 1 ns;

        assert signed(d) = to_signed(0, 64)
            report "Control Case 4 failed: D changed before clock edge"
            severity error;
        assert unsigned(s) = to_unsigned(0, 64)
            report "Control Case 4 failed: S changed before clock edge"
            severity error;
        assert signed(d_next) = to_signed(-10, 64)
            report "Control Case 4 failed: wrong d_next"
            severity error;
        assert unsigned(s_next) = to_unsigned(100, 64)
            report "Control Case 4 failed: wrong s_next"
            severity error;

        wait until rising_edge(clk);
        en <= '0';
        wait for 1 ns;

        assert signed(d) = to_signed(-10, 64)
            report "Control Case 4 failed: D not updated after clock"
            severity error;
        assert unsigned(s) = to_unsigned(100, 64)
            report "Control Case 4 failed: S not updated after clock"
            severity error;
        report "Control Case 4 passed: d_next/s_next behaviour"
            severity note;
        report
            "All D/S independent verification cases passed"
            severity note;

        wait;

    end process;

end Behavioral;