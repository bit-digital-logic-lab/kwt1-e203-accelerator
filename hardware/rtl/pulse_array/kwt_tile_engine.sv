// 单个 8×16 输出块的控制器：串行读入 K 向量，发射脉冲并写回结果。
// A 按 [K][8]、B 按 [K][16]、输出按 [8][16] 的小端 16 bit 字排列。
module kwt_tile_engine (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          start,
    input  logic [6:0]    k_len,
    input  logic [7:0]    row_mask,
    input  logic [15:0]   col_mask,
    input  logic          relu_en,
    input  logic [511:0]  bias_data,
    output logic          busy,
    output logic          done,

    input  logic          a_host_en,
    input  logic          a_host_we,
    input  logic [9:0]    a_host_addr,
    input  logic [15:0]   a_host_wdata,
    output logic [15:0]   a_host_rdata,
    input  logic          b_host_en,
    input  logic          b_host_we,
    input  logic [9:0]    b_host_addr,
    input  logic [15:0]   b_host_wdata,
    output logic [15:0]   b_host_rdata,
    input  logic          out_host_en,
    input  logic          out_host_we,
    input  logic [9:0]    out_host_addr,
    input  logic [15:0]   out_host_wdata,
    output logic [15:0]   out_host_rdata
);
    localparam int ROWS = 8;
    localparam int COLS = 16;
    localparam int DRAIN_CYCLES = ROWS + COLS + 6 + 2;

    typedef enum logic [2:0] {
        IDLE, CLEAR, READ, LAUNCH, DRAIN, WRITE_LO, WRITE_HI, FINISH
    } state_t;
    state_t state;

    logic [6:0] k_index;
    logic [4:0] read_step;
    logic [5:0] drain_step;
    logic [6:0] write_index;
    logic [127:0] a_lanes;
    logic [255:0] b_lanes;
    logic [4095:0] acc_data;
    logic [127:0] acc_update;
    logic [31:0] write_result;
    logic [15:0] a_rdata, b_rdata, out_rdata;
    logic [15:0] out_wdata;
    logic [9:0] a_addr, b_addr, out_addr;
    logic a_en, b_en, out_en, out_we;
    logic host_access;

    // 空闲及完成脉冲期间开放主机访问，运行期间缓存由控制器独占。
    assign host_access = (state == IDLE) || (state == FINISH);
    assign busy = !host_access;
    assign done = (state == FINISH);
    assign a_en = host_access ? a_host_en : (state == READ && read_step < 8);
    assign b_en = host_access ? b_host_en : (state == READ && read_step < 16);
    assign out_en = host_access ? out_host_en :
                    (state == WRITE_LO || state == WRITE_HI);
    assign out_we = host_access ? out_host_we :
                    (state == WRITE_LO || state == WRITE_HI);
    assign a_addr = host_access ? a_host_addr : 10'(k_index * 8 + read_step);
    assign b_addr = host_access ? b_host_addr : 10'(k_index * 16 + read_step);
    assign out_addr = host_access ? out_host_addr :
                      10'({write_index, (state == WRITE_HI)});
    assign out_wdata = host_access ? out_host_wdata :
                       (state == WRITE_HI) ? write_result[31:16] :
                       write_result[15:0];
    assign a_host_rdata = a_rdata;
    assign b_host_rdata = b_rdata;
    assign out_host_rdata = out_rdata;

    // 三块单口同步 SRAM：输入各一块，结果一块。
    kwt_tile_buffers buffers (
        .clk(clk),
        .a_en(a_en), .a_we(host_access && a_host_we),
        .a_addr(a_addr), .a_wdata(a_host_wdata), .a_rdata(a_rdata),
        .b_en(b_en), .b_we(host_access && b_host_we),
        .b_addr(b_addr), .b_wdata(b_host_wdata), .b_rdata(b_rdata),
        .out_en(out_en), .out_we(out_we),
        .out_addr(out_addr), .out_wdata(out_wdata), .out_rdata(out_rdata)
    );

    // 计算阵列在 LAUNCH 状态接收一个 K 序号对应的 8+16 个输入。
    kwt_pulse_array #(.ROWS(ROWS), .COLS(COLS),
                      .DATA_W(16), .ACC_W(32), .FORMAT(0)) array (
        .clk(clk), .rst_n(rst_n), .acc_clear(state == CLEAR),
        .launch(state == LAUNCH),
        .a_data(a_lanes), .a_mask(row_mask),
        .b_data(b_lanes), .b_mask(col_mask),
        .acc_data(acc_data), .acc_update(acc_update)
    );

    // 每列一份偏置；屏蔽的行列始终写零。
    logic [31:0] processed_result;
    kwt_postprocess postprocess (
        .acc_data(acc_data[write_index*32 +: 32]),
        .bias_data(bias_data[write_index[3:0]*32 +: 32]),
        .relu_en(relu_en), .result(processed_result)
    );
    assign write_result = (row_mask[write_index[6:4]] &&
                           col_mask[write_index[3:0]]) ? processed_result : 32'b0;

    // READ 第 0～15 拍发出地址，第 1～16 拍接收同步读数据。
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            k_index <= '0;
            read_step <= '0;
            drain_step <= '0;
            write_index <= '0;
            a_lanes <= '0;
            b_lanes <= '0;
        end else begin
            case (state)
                IDLE: if (start) begin
                    k_index <= '0;
                    state <= CLEAR;
                end

                CLEAR: begin
                    read_step <= '0;
                    if (k_len == 0)
                        state <= WRITE_LO;
                    else
                        state <= READ;
                    write_index <= '0;
                end

                READ: begin
                    if (read_step > 0 && read_step <= 8)
                        a_lanes[(read_step-1)*16 +: 16] <= a_rdata;
                    if (read_step > 0 && read_step <= 16)
                        b_lanes[(read_step-1)*16 +: 16] <= b_rdata;

                    if (read_step == 16)
                        state <= LAUNCH;
                    else
                        read_step <= read_step + 1'b1;
                end

                LAUNCH: begin
                    drain_step <= '0;
                    state <= DRAIN;
                end

                DRAIN: if (drain_step == 6'(DRAIN_CYCLES-1)) begin
                    if (k_index + 1'b1 == k_len) begin
                        write_index <= '0;
                        state <= WRITE_LO;
                    end else begin
                        k_index <= k_index + 1'b1;
                        read_step <= '0;
                        state <= READ;
                    end
                end else begin
                    drain_step <= drain_step + 1'b1;
                end

                WRITE_LO: state <= WRITE_HI;

                WRITE_HI: if (write_index == 127) begin
                    state <= FINISH;
                end else begin
                    write_index <= write_index + 1'b1;
                    state <= WRITE_LO;
                end

                FINISH: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
