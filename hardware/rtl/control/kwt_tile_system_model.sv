// One-tile control model. cmd_* is an abstract CPU command, bus_* is a
// simulation-only 128-bit memory interface. Neither is a SoC pin definition.
module kwt_tile_system_model #(
    parameter int EXT_MEM_BYTES = 512 * 1024
) (
    input  logic clk, rst_n,
    input  logic cmd_valid,
    output logic cmd_ready,
    input  logic [31:0] cmd_a_addr, cmd_b_addr, cmd_bias_addr, cmd_out_addr,
    input  logic [6:0] cmd_k_len,
    input  logic [7:0] cmd_row_mask,
    input  logic [15:0] cmd_col_mask,
    input  logic cmd_relu_en,
    output logic rsp_valid,
    input  logic rsp_ready,
    output logic [1:0] rsp_status, // 0=OK, 1=command, 2=DMA

    output logic bus_valid, bus_write,
    input  logic bus_ready,
    output logic [31:0] bus_addr,
    output logic [127:0] bus_wdata,
    output logic [15:0] bus_wstrb,
    input  logic bus_rvalid,
    input  logic [127:0] bus_rdata
);
    typedef enum logic [3:0] {
        IDLE, LOAD_A, WAIT_A, LOAD_B, WAIT_B, LOAD_BIAS, WAIT_BIAS,
        BIAS_REQ, BIAS_CAPTURE, RUN_TILE, WAIT_TILE,
        STORE_OUT, WAIT_STORE, RESPONSE
    } state_t;
    state_t state;

    logic [31:0] a_addr_q, b_addr_q, bias_addr_q, out_addr_q;
    logic [6:0] k_len_q;
    logic [7:0] row_mask_q;
    logic [15:0] col_mask_q;
    logic relu_q;
    logic [511:0] bias_q;
    logic [5:0] bias_index;
    logic [1:0] status_q;

    logic dma_start, dma_writeback, dma_busy, dma_done, dma_error;
    logic [1:0] dma_bank;
    logic [31:0] dma_ext_addr;
    logic [9:0] dma_sram_addr;
    logic [15:0] dma_words;
    logic da_en, da_we, db_en, db_we, do_en, do_we;
    logic [9:0] da_addr, db_addr, do_addr;
    logic [15:0] da_wdata, db_wdata, do_wdata;
    logic [15:0] a_rdata, b_rdata, out_rdata;
    logic tile_busy, tile_done;
    logic bias_read;

    assign cmd_ready = state == IDLE;
    assign rsp_valid = state == RESPONSE;
    assign rsp_status = status_q;
    // The output SRAM has a registered read port. Prime word zero, then
    // request the next word while capturing the previous read result.
    assign bias_read = state == BIAS_REQ ||
                       (state == BIAS_CAPTURE && bias_index != 6'd31);
    assign dma_start = state == LOAD_A || state == LOAD_B ||
                       state == LOAD_BIAS || state == STORE_OUT;
    assign dma_writeback = state == STORE_OUT;
    assign dma_bank = (state == LOAD_A || state == WAIT_A) ? 2'd0 :
                      (state == LOAD_B || state == WAIT_B) ? 2'd1 : 2'd2;
    assign dma_ext_addr = (state == LOAD_A || state == WAIT_A) ? a_addr_q :
                          (state == LOAD_B || state == WAIT_B) ? b_addr_q :
                          (state == LOAD_BIAS || state == WAIT_BIAS) ? bias_addr_q : out_addr_q;
    assign dma_sram_addr = (state == LOAD_BIAS || state == WAIT_BIAS) ? 10'd256 : 10'd0;
    assign dma_words = (state == LOAD_A || state == WAIT_A) ? {6'b0, k_len_q, 3'b0} :
                       (state == LOAD_B || state == WAIT_B) ? {5'b0, k_len_q, 4'b0} :
                       (state == LOAD_BIAS || state == WAIT_BIAS) ? 16'd32 : 16'd256;

    kwt_dma #(.EXT_MEM_BYTES(EXT_MEM_BYTES)) dma (
        .clk(clk), .rst_n(rst_n), .start(dma_start),
        .writeback(dma_writeback), .bank(dma_bank),
        .ext_byte_addr(dma_ext_addr), .sram_word_addr(dma_sram_addr),
        .word_count(dma_words), .busy(dma_busy), .done(dma_done), .error(dma_error),
        .bus_valid(bus_valid), .bus_write(bus_write), .bus_ready(bus_ready),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_wstrb(bus_wstrb),
        .bus_rvalid(bus_rvalid), .bus_rdata(bus_rdata),
        .a_en(da_en), .a_we(da_we), .a_addr(da_addr), .a_wdata(da_wdata), .a_rdata(a_rdata),
        .b_en(db_en), .b_we(db_we), .b_addr(db_addr), .b_wdata(db_wdata), .b_rdata(b_rdata),
        .out_en(do_en), .out_we(do_we), .out_addr(do_addr),
        .out_wdata(do_wdata), .out_rdata(out_rdata)
    );

    kwt_pingpong_cacc_engine tile (
        .clk(clk), .rst_n(rst_n), .start(state == RUN_TILE),
        .compute_page(1'b0), .load_page(1'b0), .load_ready(),
        .k_len(k_len_q), .row_mask(row_mask_q), .col_mask(col_mask_q),
        .relu_en(relu_q), .bias_data(bias_q), .busy(tile_busy), .done(tile_done),
        .cacc_collision(),
        .a_host_en(da_en), .a_host_we(da_we), .a_host_addr(da_addr),
        .a_host_wdata(da_wdata), .a_host_rdata(a_rdata),
        .b_host_en(db_en), .b_host_we(db_we), .b_host_addr(db_addr),
        .b_host_wdata(db_wdata), .b_host_rdata(b_rdata),
        .out_host_en(bias_read || do_en), .out_host_we(do_we),
        .out_host_addr(bias_read ?
                       (10'd256 + {4'b0, bias_index} +
                        (state == BIAS_CAPTURE ? 10'd1 : 10'd0)) : do_addr),
        .out_host_wdata(do_wdata), .out_host_rdata(out_rdata)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            a_addr_q <= '0;
            b_addr_q <= '0;
            bias_addr_q <= '0;
            out_addr_q <= '0;
            k_len_q <= '0;
            row_mask_q <= '0;
            col_mask_q <= '0;
            relu_q <= 1'b0;
            bias_q <= '0;
            bias_index <= '0;
            status_q <= '0;
        end else begin
            case (state)
                IDLE: if (cmd_valid) begin
                    a_addr_q <= cmd_a_addr;
                    b_addr_q <= cmd_b_addr;
                    bias_addr_q <= cmd_bias_addr;
                    out_addr_q <= cmd_out_addr;
                    k_len_q <= cmd_k_len;
                    row_mask_q <= cmd_row_mask;
                    col_mask_q <= cmd_col_mask;
                    relu_q <= cmd_relu_en;
                    status_q <= 0;
                    if (cmd_k_len == 0 || cmd_k_len > 64 ||
                        cmd_row_mask == 0 || cmd_col_mask == 0 ||
                        |cmd_a_addr[3:0] || |cmd_b_addr[3:0] ||
                        |cmd_bias_addr[3:0] || |cmd_out_addr[3:0]) begin
                        status_q <= 1;
                        state <= RESPONSE;
                    end else state <= LOAD_A;
                end
                LOAD_A: state <= WAIT_A;
                WAIT_A: if (dma_done) begin
                    if (dma_error) begin status_q <= 2; state <= RESPONSE; end
                    else state <= LOAD_B;
                end
                LOAD_B: state <= WAIT_B;
                WAIT_B: if (dma_done) begin
                    if (dma_error) begin status_q <= 2; state <= RESPONSE; end
                    else state <= LOAD_BIAS;
                end
                LOAD_BIAS: state <= WAIT_BIAS;
                WAIT_BIAS: if (dma_done) begin
                    if (dma_error) begin status_q <= 2; state <= RESPONSE; end
                    else begin bias_index <= 0; state <= BIAS_REQ; end
                end
                BIAS_REQ: state <= BIAS_CAPTURE;
                BIAS_CAPTURE: begin
                    bias_q[bias_index*16 +: 16] <= out_rdata;
                    if (bias_index == 31) state <= RUN_TILE;
                    else bias_index <= bias_index + 1'b1;
                end
                RUN_TILE: state <= WAIT_TILE;
                WAIT_TILE: if (tile_done) state <= STORE_OUT;
                STORE_OUT: state <= WAIT_STORE;
                WAIT_STORE: if (dma_done) begin
                    if (dma_error) status_q <= 2;
                    state <= RESPONSE;
                end
                RESPONSE: if (rsp_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
