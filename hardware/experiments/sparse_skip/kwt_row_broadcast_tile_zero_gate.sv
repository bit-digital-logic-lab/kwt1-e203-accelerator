// Eight row lanes share one B word per cycle. Each lane owns one output row;
// its 16 columns are accumulated in a separate 1R1W partial-sum bank.
// A and B buffering, the task scheduler, and output DMA are outside this core.
module kwt_row_broadcast_tile_zero_gate #(
    parameter int ROWS = 8,
    parameter int COLS = 16,
    parameter int DATA_W = 16,
    parameter int ACC_W = 32,
    parameter int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter int COL_W = (COLS <= 2) ? 1 : $clog2(COLS)
) (
    input  logic clk, rst_n,
    input  logic clear,
    input  logic in_valid,
    input  logic [ROWS-1:0] row_mask,
    input  logic [ROWS*DATA_W-1:0] a_rows,
    input  logic [DATA_W-1:0] b_word,
    input  logic [COL_W-1:0] in_col,
    input  logic dump_req,
    input  logic [ROW_W-1:0] dump_row,
    input  logic [COL_W-1:0] dump_col,
    input  logic [ACC_W-1:0] dump_bias,
    output logic dump_valid,
    output logic [ACC_W-1:0] dump_data
);
    logic [ROWS-1:0] nonzero_rows;
    for (genvar r = 0; r < ROWS; r++) begin : g_nonzero
        assign nonzero_rows[r] = (a_rows[r*DATA_W +: DATA_W] != '0);
    end
    logic [ROWS-1:0] product_valid_q;
    logic [COL_W-1:0] product_col_q;
    logic signed [2*DATA_W-1:0] product_q [0:ROWS-1];
    logic [ROWS*ACC_W-1:0] bank_read_bus;
    logic [ROWS-1:0] bank_valid_bus;
    logic [ROW_W-1:0] dump_row_q;
    logic [COL_W-1:0] dump_col_q;
    logic [ACC_W-1:0] dump_bias_q;
    logic [ACC_W-1:0] dump_sum;

    assign dump_sum = bank_valid_bus[dump_row_q] ?
                      bank_read_bus[dump_row_q*ACC_W +: ACC_W] : '0;
    assign dump_data = dump_sum + dump_bias_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_valid_q <= '0;
            product_col_q <= '0;
            for (int r = 0; r < ROWS; r++) product_q[r] <= '0;
        end else if (clear) begin
            product_valid_q <= '0;
        end else begin
            product_valid_q <= {ROWS{in_valid && (b_word != '0)}} & row_mask & nonzero_rows;
            product_col_q <= in_col;
            if (in_valid && (b_word != '0)) begin
                for (int r = 0; r < ROWS; r++)
                    if (row_mask[r] && nonzero_rows[r]) product_q[r] <= $signed(a_rows[r*DATA_W +: DATA_W]) *
                                    $signed(b_word);
            end
        end
    end

    for (genvar r = 0; r < ROWS; r++) begin : g_psum_bank
        logic signed [ACC_W-1:0] psum_mem [0:COLS-1];
        logic [COLS-1:0] valid_bits;
        logic signed [ACC_W-1:0] read_data;
        logic pending, pending_old_valid;
        logic [COL_W-1:0] pending_col;
        logic signed [2*DATA_W-1:0] pending_product;
        logic forward_valid;
        logic signed [ACC_W-1:0] forward_value;
        logic signed [ACC_W-1:0] pending_sum;

        assign pending_sum =
            (forward_valid ? forward_value :
             pending_old_valid ? read_data : '0) + ACC_W'(pending_product);
        assign bank_read_bus[r*ACC_W +: ACC_W] = read_data;
        assign bank_valid_bus[r] = valid_bits[dump_col_q];

        // Reset only validity; the data array remains SRAM-inferable.
        always_ff @(posedge clk) begin
            if (!clear) begin
                if (pending) psum_mem[pending_col] <= pending_sum;
                if (product_valid_q[r]) read_data <= psum_mem[product_col_q];
                else if (dump_req && dump_row == ROW_W'(r))
                    read_data <= psum_mem[dump_col];
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_bits <= '0;
                pending <= 1'b0;
                pending_col <= '0;
                pending_product <= '0;
                pending_old_valid <= 1'b0;
                forward_valid <= 1'b0;
                forward_value <= '0;
            end else if (clear) begin
                valid_bits <= '0;
                pending <= 1'b0;
                forward_valid <= 1'b0;
            end else begin
                if (pending) valid_bits[pending_col] <= 1'b1;
                pending <= product_valid_q[r];
                forward_valid <= product_valid_q[r] && pending &&
                                 (product_col_q == pending_col);
                if (product_valid_q[r]) begin
                    pending_col <= product_col_q;
                    pending_product <= product_q[r];
                    pending_old_valid <= valid_bits[product_col_q];
                    if (pending && product_col_q == pending_col)
                        forward_value <= pending_sum;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dump_valid <= 1'b0;
            dump_row_q <= '0;
            dump_col_q <= '0;
            dump_bias_q <= '0;
        end else if (clear) begin
            dump_valid <= 1'b0;
        end else begin
            dump_valid <= dump_req;
            if (dump_req) begin
                dump_row_q <= dump_row;
                dump_col_q <= dump_col;
                dump_bias_q <= dump_bias;
            end
        end
    end
endmodule
