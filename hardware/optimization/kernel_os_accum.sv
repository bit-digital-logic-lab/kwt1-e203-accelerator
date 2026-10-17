// Experimental output-stationary accumulator: one 32-bit register and adder
// per output. This is a microkernel, not a replacement for the tile engine.
module kernel_os_accum #(
    parameter int ROWS = 2,
    parameter int COLS = 3,
    parameter int DATA_W = 16,
    parameter int ACC_W = 32
) (
    input  logic clk, rst_n, clear,
    input  logic [ROWS*COLS*2*DATA_W-1:0] product_data,
    input  logic [ROWS*COLS-1:0] product_valid,
    output logic [ROWS*COLS*ACC_W-1:0] sum_data
);
    for (genvar i = 0; i < ROWS*COLS; i++) begin : g_output
        logic signed [ACC_W-1:0] sum_q;
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) sum_q <= '0;
            else if (clear) sum_q <= '0;
            else if (product_valid[i])
                sum_q <= sum_q + ACC_W'($signed(product_data[i*2*DATA_W +: 2*DATA_W]));
        end
        assign sum_data[i*ACC_W +: ACC_W] = sum_q;
    end
endmodule
