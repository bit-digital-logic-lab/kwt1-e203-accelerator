// One 1R1W partial-sum bank per array row. A row emits at most one product
// per cycle; the next cycle writes the updated sum to that product's column.
module kwt_cacc #(
    parameter int ROWS = 8,
    parameter int COLS = 16,
    parameter int DATA_W = 16,
    parameter int ACC_W = 32,
    parameter int COL_W = (COLS <= 2) ? 1 : $clog2(COLS),
    parameter int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS)
) (
    input logic clk, rst_n,
    input logic clear,
    input logic [ROWS*COLS*2*DATA_W-1:0] product_data,
    input logic [ROWS*COLS-1:0] product_valid,
    output logic collision_seen,
    input logic dump_req,
    input logic [ROW_W-1:0] dump_row,
    input logic [COL_W-1:0] dump_col,
    output logic dump_valid,
    output logic [ACC_W-1:0] dump_data
);
    logic row_fire [0:ROWS-1];
    logic [COL_W-1:0] row_col [0:ROWS-1];
    logic signed [2*DATA_W-1:0] row_product [0:ROWS-1];
    logic row_collision [0:ROWS-1];
    logic [ROWS*ACC_W-1:0] bank_read_bus;
    logic [ROWS-1:0] bank_valid_bus;
    logic [ROW_W-1:0] dump_row_q;
    logic [COL_W-1:0] dump_col_q;

    assign dump_data = bank_valid_bus[dump_row_q] ?
                       bank_read_bus[dump_row_q*ACC_W +: ACC_W] : '0;

    for (genvar r = 0; r < ROWS; r++) begin : g_bank
        logic signed [ACC_W-1:0] psum_mem [0:COLS-1];
        logic [COLS-1:0] valid_bits;
        logic signed [ACC_W-1:0] read_data;
        logic pending, pending_old_valid;
        logic [COL_W-1:0] pending_col;
        logic signed [2*DATA_W-1:0] pending_product;
        logic forward_valid;
        logic signed [ACC_W-1:0] forward_value;
        logic signed [ACC_W-1:0] pending_sum;

        always_comb begin
            row_fire[r] = 1'b0;
            row_col[r] = '0;
            row_product[r] = '0;
            row_collision[r] = 1'b0;
            for (int c = 0; c < COLS; c++) begin
                if (product_valid[r*COLS+c]) begin
                    if (row_fire[r]) row_collision[r] = 1'b1;
                    row_fire[r] = 1'b1;
                    row_col[r] = COL_W'(c);
                    row_product[r] = $signed(product_data[(r*COLS+c)*2*DATA_W +: 2*DATA_W]);
                end
            end
        end

        assign pending_sum =
            (forward_valid ? forward_value :
             pending_old_valid ? read_data : '0) + ACC_W'(pending_product);
        assign bank_read_bus[r*ACC_W +: ACC_W] = read_data;
        assign bank_valid_bus[r] = valid_bits[dump_col_q];

        // Keep the data array reset-free so synthesis can infer a small SRAM.
        // The per-column valid bits make stale data invisible after clear.
        always_ff @(posedge clk) begin
            if (!clear) begin
                if (pending) psum_mem[pending_col] <= pending_sum;
                if (row_fire[r]) read_data <= psum_mem[row_col[r]];
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
                pending <= row_fire[r];
                forward_valid <= row_fire[r] && pending && (row_col[r] == pending_col);
                if (row_fire[r]) begin
                    pending_col <= row_col[r];
                    pending_product <= row_product[r];
                    pending_old_valid <= valid_bits[row_col[r]];
                    if (pending && row_col[r] == pending_col)
                        forward_value <= pending_sum;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            collision_seen <= 1'b0;
            dump_valid <= 1'b0;
            dump_row_q <= '0;
            dump_col_q <= '0;
        end else if (clear) begin
            collision_seen <= 1'b0;
            dump_valid <= 1'b0;
        end else begin
            dump_valid <= dump_req;
            if (dump_req) begin
                dump_row_q <= dump_row;
                dump_col_q <= dump_col;
            end
            for (int r = 0; r < ROWS; r++)
                if (row_collision[r]) collision_seen <= 1'b1;
        end
    end
endmodule
