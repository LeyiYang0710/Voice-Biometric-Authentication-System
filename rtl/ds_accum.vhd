-- ds_accum.vhd
-- D/S accumulator for the Quality Monitor: D = sum(x), S = sum(x*x).
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ds_accum is 
    generic (
        DATA_WIDTH : positive := 16;    -- sample data width
        ACC_WIDTH  : positive := 64     -- accumulator's width
    );
    port (
        clk     : in std_logic;
        rst     : in std_logic;
        clear   : in std_logic;
        en      : in std_logic;            -- = 1 when there is an accepted sample in the curr cc
        x       : in std_logic_vector(DATA_WIDTH - 1 downto 0);    -- sample
        d       : out std_logic_vector(ACC_WIDTH - 1 downto 0);    -- curr sum(x)
        s       : out std_logic_vector(ACC_WIDTH-1 downto 0);      -- curr sum(x*x)
        d_next  : out std_logic_vector(ACC_WIDTH-1 downto 0);      -- d after this cycle's sample
        s_next  : out std_logic_vector(ACC_WIDTH-1 downto 0)       -- s after this cycle's sample
    );
end ds_accum;

architecture arch of ds_accum is
    signal sig_d    : signed(ACC_WIDTh - 1 downto 0) := (others => '0');     -- the reg for d
    signal sig_s      : unsigned(ACC_WIDTH-1 downto 0)   := (others => '0'); -- the reg for s
    signal sig_sq     : unsigned(2*DATA_WIDTH-1 downto 0);                   -- square of the curr x
    signal sig_d_next : signed(ACC_WIDTH-1 downto 0);                        
    signal sig_s_next : unsigned(ACC_WIDTH-1 downto 0);
    
    begin
    sig_sq     <= unsigned(signed(x) * signed(x));          -- do type conversion to complete the multiplication
    sig_d_next <= sig_d + resize(signed(x), ACC_WIDTH);     -- extend the x to the same width as sig_d and sum up
    sig_s_next <= sig_s + resize(sig_sq, ACC_WIDTH);        
    
    
    process (clk)
    begin
        if rising_edge(clk) then
            if rst = '0' or clear = '1' then
                sig_d <= (others => '0');
                sig_s <= (others => '0');
            else
                if en = '1' then        -- when there is an accepted sample in the curr cc
                    sig_d <= sig_d_next;
                    sig_s <= sig_s_next;
                end if;
            end if;
        end if;
    end process;    
    
    d      <= std_logic_vector(sig_d);
    s      <= std_logic_vector(sig_s);
    d_next <= std_logic_vector(sig_d_next);
    s_next <= std_logic_vector(sig_s_next);
    
end arch;