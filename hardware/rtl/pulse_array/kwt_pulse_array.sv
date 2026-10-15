// 整数和浮点 PE 共用的参数化二维脉冲矩阵。
// A 向右传播，B 向下传播；边界错拍使相同 K 序号的数据在 PE 内相遇。
module kwt_pulse_array #(
    parameter int ROWS = 8,
    parameter int COLS = 16,
    parameter int DATA_W = 32,
    parameter int ACC_W = 32,
    parameter int FORMAT = 0
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         acc_clear,
    input  logic                         launch,

    input  logic [ROWS*DATA_W-1:0]       a_data,
    input  logic [ROWS-1:0]              a_mask,

    input  logic [COLS*DATA_W-1:0]       b_data,
    input  logic [COLS-1:0]              b_mask,

    output logic [ROWS*COLS*ACC_W-1:0]  acc_data,
    output logic [ROWS*COLS-1:0]        acc_update
);
    // PE 之间传递的数据和有效标志。
    logic [DATA_W-1:0] a_link [0:ROWS-1][0:COLS];
    logic              a_valid [0:ROWS-1][0:COLS];

    logic [DATA_W-1:0] b_link [0:ROWS][0:COLS-1];
    logic              b_valid [0:ROWS][0:COLS-1];

    // A 的第 r 行在左侧延迟 r 拍注入。
    for (genvar r = 0; r < ROWS; r++) begin : g_a_edge
        if (r == 0) begin : g_direct
            assign a_link[r][0] = a_data[r*DATA_W +: DATA_W];
            assign a_valid[r][0] = launch && a_mask[r];
        end else begin : g_delay
            // 非首行通过移位寄存器形成行错拍。
            logic [DATA_W-1:0] word [0:r-1];
            logic              valid [0:r-1];

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

    // B 的第 c 列在上侧延迟 c 拍注入。
    for (genvar c = 0; c < COLS; c++) begin : g_b_edge
        if (c == 0) begin : g_direct
            assign b_link[0][c] = b_data[c*DATA_W +: DATA_W];
            assign b_valid[0][c] = launch && b_mask[c];
        end else begin : g_delay
            // 非首列通过移位寄存器形成列错拍。
            logic [DATA_W-1:0] word [0:c-1];
            logic              valid [0:c-1];

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

    // 实例化 ROWS × COLS 个 PE，并连接相邻 PE 的传播端口。
    for (genvar r = 0; r < ROWS; r++) begin : g_rows
        for (genvar c = 0; c < COLS; c++) begin : g_cols
            kwt_pe #(.DATA_W(DATA_W), .ACC_W(ACC_W), .FORMAT(FORMAT)) pe (
                .clk(clk), .rst_n(rst_n), .acc_clear(acc_clear),
                .a_in(a_link[r][c]), .a_valid_in(a_valid[r][c]),
                .b_in(b_link[r][c]), .b_valid_in(b_valid[r][c]),
                .a_out(a_link[r][c+1]), .a_valid_out(a_valid[r][c+1]),
                .b_out(b_link[r+1][c]), .b_valid_out(b_valid[r+1][c]),
                .acc(acc_data[(r*COLS+c)*ACC_W +: ACC_W]),
                .acc_update(acc_update[r*COLS+c])
            );
        end
    end
endmodule
