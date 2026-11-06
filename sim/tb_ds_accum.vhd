-- tb_ds_accum.vhd
-- Testbench for ds_accum.

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity ds_accum_TB is
end ds_accum_TB;

architecture Behavioral of ds_accum_TB is
    component ds_accum is
        generic (
            DATA_WIDTH : positive := 16;
            ACC_WIDTH  : positive := 64
        );
        port (
            clk     : in  std_logic;
            rst     : in  std_logic;
            clear   : in  std_logic;
            en      : in  std_logic;
            x       : in  std_logic_vector(DATA_WIDTH-1 downto 0);
            d       : out std_logic_vector(ACC_WIDTH-1 downto 0);
            s       : out std_logic_vector(ACC_WIDTH-1 downto 0);
            d_next  : out std_logic_vector(ACC_WIDTH-1 downto 0);
            s_next  : out std_logic_vector(ACC_WIDTH-1 downto 0)
        );
    end component;

    signal sig_clk    : std_logic := '0';
    signal sig_rst    : std_logic := '0';
    signal sig_clear  : std_logic := '0';
    signal sig_en     : std_logic := '0';
    signal sig_x      : std_logic_vector(15 downto 0) := (others => '0');
    signal sig_d      : std_logic_vector(63 downto 0) := (others => '0');
    signal sig_s      : std_logic_vector(63 downto 0) := (others => '0');
    signal sig_d_next : std_logic_vector(63 downto 0) := (others => '0');
    signal sig_s_next : std_logic_vector(63 downto 0) := (others => '0');
begin
    -- Set the global clk
    sig_clk <= not sig_clk after 10 ns;

    process is
        -- Accept one sample for one clock, then en = '0' for one clock
        -- carrying a junk value that must be ignored.
        procedure send_sample (val : in integer) is
        begin
            sig_x  <= std_logic_vector(to_signed(val, 16));
            sig_en <= '1';
            wait until rising_edge(sig_clk);
            sig_x  <= std_logic_vector(to_signed(1234, 16));
            sig_en <= '0';
            wait until rising_edge(sig_clk);
        end procedure;

        -- One-cycle clear pulse
        procedure do_clear is
        begin
            sig_clear <= '1';
            wait until rising_edge(sig_clk);
            sig_clear <= '0';
            wait until rising_edge(sig_clk);
        end procedure;
    begin
        sig_rst <= '0';
        wait for 100 ns;
        sig_rst <= '1';
        wait until rising_edge(sig_clk);

        -- Case 1: positive samples 10, 20, 30  :  D = 60, S = 1400
        send_sample(10);
        send_sample(20);
        send_sample(30);
        wait for 1 ns;
        assert signed(sig_d) = to_signed(60, 64);
        assert unsigned(sig_s) = to_unsigned(1400, 64);

        -- Clear
        do_clear;
        wait for 1 ns;
        assert signed(sig_d) = to_signed(0, 64);
        assert unsigned(sig_s) = to_unsigned(0, 64);

        -- Case 2: mixed signs 5, -8, 12, -3  :  D = 6, S = 242
        send_sample(5);
        send_sample(-8);
        send_sample(12);
        send_sample(-3);
        wait for 1 ns;
        assert signed(sig_d) = to_signed(6, 64);
        assert unsigned(sig_s) = to_unsigned(242, 64);

        -- Stop
        wait for 1 us;
    end process;

    test: ds_accum port map (
        clk    => sig_clk,
        rst    => sig_rst,
        clear  => sig_clear,
        en     => sig_en,
        x      => sig_x,
        d      => sig_d,
        s      => sig_s,
        d_next => sig_d_next,
        s_next => sig_s_next
    );

end Behavioral;