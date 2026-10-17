// Systolic operand propagation. Each PE holds only a registered product;
// cross-K partial sums live in kwt_cacc, outside this array.
module kwt_product_array #(
    parameter int ROWS = 8,
    parameter int COLS = 16,
    parameter int DATA_W = 16
) (
    input  logic clk, rst_n,
    input  logic launch,
    input  logic [ROWS*DATA_W-1:0] a_data,
    input  logic [ROWS-1:0] a_mask,
    input  logic [COLS*DATA_W-1:0] b_data,
    input  logic [COLS-1:0] b_mask,
    output logic [ROWS*COLS*2*DATA_W-1:0] product_data,
    output logic [ROWS*COLS-1:0] product_valid
);
    logic [DATA_W-1:0] a_link [0:ROWS-1][0:COLS];
    logic [DATA_W-1:0] b_link [0:ROWS][0:COLS-1];
    logic a_valid [0:ROWS-1][0:COLS];
    logic b_valid [0:ROWS][0:COLS-1];

    for (genvar r = 0; r < ROWS; r++) begin : g_a_edge
        if (r == 0) begin : g_direct
            assign a_link[r][0] = a_data[r*DATA_W +: DATA_W];
            assign a_valid[r][0] = launch && a_mask[r];
        end else begin : g_skew
            logic [DATA_W-1:0] word [0:r-1];
            logic valid [0:r-1];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < r; i++) begin
                        word[i] <= '0;
                        valid[i] <= 1'b0;
                    end
                end else begin
                    word[0] <= a_data[r*DATA_W +: DATA_W];
                    valid[0] <= launch && a_mask[r];
                    for (int i = 1; i < r; i++) begin
                        word[i] <= word[i-1];
                        valid[i] <= valid[i-1];
                    end
                end
            end
            assign a_link[r][0] = word[r-1];
            assign a_valid[r][0] = valid[r-1];
        end
    end

    for (genvar c = 0; c < COLS; c++) begin : g_b_edge
        if (c == 0) begin : g_direct
            assign b_link[0][c] = b_data[c*DATA_W +: DATA_W];
            assign b_valid[0][c] = launch && b_mask[c];
        end else begin : g_skew
            logic [DATA_W-1:0] word [0:c-1];
            logic valid [0:c-1];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < c; i++) begin
                        word[i] <= '0;
                        valid[i] <= 1'b0;
                    end
                end else begin
                    word[0] <= b_data[c*DATA_W +: DATA_W];
                    valid[0] <= launch && b_mask[c];
                    for (int i = 1; i < c; i++) begin
                        word[i] <= word[i-1];
                        valid[i] <= valid[i-1];
                    end
                end
            end
            assign b_link[0][c] = word[c-1];
            assign b_valid[0][c] = valid[c-1];
        end
    end

    for (genvar r = 0; r < ROWS; r++) begin : g_row
        for (genvar c = 0; c < COLS; c++) begin : g_col
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    a_link[r][c+1] <= '0;
                    b_link[r+1][c] <= '0;
                    a_valid[r][c+1] <= 1'b0;
                    b_valid[r+1][c] <= 1'b0;
                    product_data[(r*COLS+c)*2*DATA_W +: 2*DATA_W] <= '0;
                    product_valid[r*COLS+c] <= 1'b0;
                end else begin
                    a_link[r][c+1] <= a_link[r][c];
                    b_link[r+1][c] <= b_link[r][c];
                    a_valid[r][c+1] <= a_valid[r][c];
                    b_valid[r+1][c] <= b_valid[r][c];
                    product_data[(r*COLS+c)*2*DATA_W +: 2*DATA_W] <=
                        $signed(a_link[r][c]) * $signed(b_link[r][c]);
                    product_valid[r*COLS+c] <= a_valid[r][c] && b_valid[r][c];
                end
            end
        end
    end
endmodule
