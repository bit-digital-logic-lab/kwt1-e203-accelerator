// COMPARISON-ONLY wrapper. It instantiates the accepted original
// kwt_row_broadcast_tile TWICE, which is exactly what the old
// hardware/experiments/streaming_mac dual-CACC task harness did to obtain two
// accumulator pages. It is used here only to measure the elaborated multiplier
// count of that two-core approach; it is NOT the new implementation.
module kwt_two_core_compare_wrapper #(
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int DATA_W = 16,
    parameter int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter int COL_W = (COLS <= 2) ? 1 : $clog2(COLS)
) (
    input  logic clk, rst_n,
    input  logic [ROWS-1:0] row_mask,
    input  logic [ROWS*DATA_W-1:0] a_rows,
    input  logic [DATA_W-1:0] b_word,
    input  logic [COL_W-1:0] in_col,
    input  logic in_valid0, in_valid1, clear0, clear1,
    input  logic dump_req0, dump_req1,
    input  logic [ROW_W-1:0] dump_row,
    input  logic [COL_W-1:0] dump_col,
    input  logic [31:0] dump_bias,
    output logic dump_valid0, dump_valid1,
    output logic [31:0] dump_data0, dump_data1
);
    kwt_row_broadcast_tile #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W)) core0 (
        .clk(clk), .rst_n(rst_n), .clear(clear0),
        .in_valid(in_valid0), .row_mask(row_mask), .a_rows(a_rows),
        .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req0), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(dump_valid0), .dump_data(dump_data0)
    );
    kwt_row_broadcast_tile #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W)) core1 (
        .clk(clk), .rst_n(rst_n), .clear(clear1),
        .in_valid(in_valid1), .row_mask(row_mask), .a_rows(a_rows),
        .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req1), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(dump_valid1), .dump_data(dump_data1)
    );
endmodule
