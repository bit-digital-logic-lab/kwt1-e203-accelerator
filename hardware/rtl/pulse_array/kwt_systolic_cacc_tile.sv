// Compute-only synthesis boundary for comparing the existing 8x16 array with
// the row-broadcast candidate. Operand SRAM and task control are excluded.
module kwt_systolic_cacc_tile #(
    parameter int ROWS = 8,
    parameter int COLS = 16,
    parameter int DATA_W = 16,
    parameter int ACC_W = 32,
    parameter int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter int COL_W = (COLS <= 2) ? 1 : $clog2(COLS)
) (
    input  logic clk, rst_n, clear, launch,
    input  logic [ROWS*DATA_W-1:0] a_data,
    input  logic [ROWS-1:0] a_mask,
    input  logic [COLS*DATA_W-1:0] b_data,
    input  logic [COLS-1:0] b_mask,
    input  logic dump_req,
    input  logic [ROW_W-1:0] dump_row,
    input  logic [COL_W-1:0] dump_col,
    input  logic [ACC_W-1:0] dump_bias,
    output logic dump_valid,
    output logic [ACC_W-1:0] dump_data,
    output logic collision_seen
);
    logic [ROWS*COLS*2*DATA_W-1:0] product_data;
    logic [ROWS*COLS-1:0] product_valid;
    logic [ACC_W-1:0] raw_dump_data, dump_bias_q;

    assign dump_data = raw_dump_data + dump_bias_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) dump_bias_q <= '0;
        else if (dump_req) dump_bias_q <= dump_bias;
    end

    kwt_product_array #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W)) array (
        .clk(clk), .rst_n(rst_n), .launch(launch),
        .a_data(a_data), .a_mask(a_mask),
        .b_data(b_data), .b_mask(b_mask),
        .product_data(product_data), .product_valid(product_valid)
    );
    kwt_cacc #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)) cacc (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .product_data(product_data), .product_valid(product_valid),
        .collision_seen(collision_seen),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_valid(dump_valid), .dump_data(raw_dump_data)
    );
endmodule
