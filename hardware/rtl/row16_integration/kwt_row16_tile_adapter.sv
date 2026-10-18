`include "kwt_memory_defs.svh"
// Single-tile 16-row adapter that plugs the 16-row compute core
// (kwt_row_broadcast_tile, ROWS=16) into the *same* NICE-like DMA system model
// used by the legacy 8x16 path: a NICE-style one-shot command descriptor
// (addresses, k_len, row/col masks) plus the shared 128-bit simulation bus and
// the 16-bit SRAM. A/W/bias are fetched through kwt_dma; the 16x16 CACC tile is
// written back through kwt_dma. This is NOT a competition SoC / System Bus
// adapter; see hardware/optimization/ROW16_E203_INTEGRATION.md for the exact
// interface gaps (row_mask width, descriptor layout, A read-port structure).
//
// Difference vs. the 8-row tile_system_model:
//   * ROWS=16 -> row_mask is 16 bits (was 8), and a 99-row M layer is covered
//     by 7 row tiles (6 full + 1 with 3 active rows), i.e. 7*16=112 tiles
//     instead of the 8-row path's 13*16=208.
//   * the core is kwt_row_broadcast_tile (one B word per cycle broadcast to 16
//     row lanes with per-row 32-bit partial-sum banks), not kwt_pulse_array.
module kwt_row16_tile_adapter #(
    parameter int EXT_MEM_BYTES = `KWT_EXT_MEM_BYTES,
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int DATA_W = 16,
    parameter int ACC_W = 32,
    parameter int A_DEPTH = 1024,   // 64 k * 16 rows
    parameter int W_DEPTH = 1024,   // 64 k * 16 cols
    parameter int O_DEPTH = 1024,
    parameter int O_ADDR_W = 10,
    parameter int BIAS_SRAM = 512   // bias words live at out_mem[512..543]
) (
    input  logic clk, rst_n,
    input  logic cmd_valid,
    output logic cmd_ready,
    input  logic [31:0] cmd_a_addr, cmd_b_addr, cmd_bias_addr, cmd_out_addr,
    input  logic [6:0] cmd_k_len,
    input  logic [ROWS-1:0] cmd_row_mask,
    input  logic [COLS-1:0] cmd_col_mask,
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
    localparam int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS);
    localparam int COL_W = (COLS <= 2) ? 1 : $clog2(COLS);

    typedef enum logic [3:0] {
        IDLE, LOAD_A, WAIT_A, LOAD_B, WAIT_B, LOAD_BIAS, WAIT_BIAS,
        CLR, RUN, SETTLE, DRAIN, STORE_OUT, WAIT_STORE, RESPONSE
    } state_t;
    state_t state;

    logic [31:0] a_addr_q, b_addr_q, bias_addr_q, out_addr_q;
    logic [6:0] k_len_q;
    logic [ROWS-1:0] row_mask_q;
    logic [COLS-1:0] col_mask_q;
    logic relu_q;
    logic [1:0] status_q;
    logic [6:0] run_k_q;
    logic [COL_W-1:0] run_c_q;
    logic [3:0] settle_q;
    logic [8:0] dreq_idx, cap_idx;

    // DMA <-> SRAM buses
    logic dma_start, dma_writeback, dma_busy, dma_done, dma_error;
    logic [1:0] dma_bank;
    logic [31:0] dma_ext_addr;
    logic [O_ADDR_W-1:0] dma_sram_addr;
    logic [15:0] dma_words;
    logic da_en, da_we, db_en, db_we, do_en, do_we;
    logic [O_ADDR_W-1:0] da_addr, db_addr, do_addr;
    logic [DATA_W-1:0] da_wdata, db_wdata, do_wdata;
    logic [DATA_W-1:0] a_rdata, b_rdata, out_rdata;

    logic [DATA_W-1:0] a_mem [0:A_DEPTH-1];
    logic [DATA_W-1:0] w_mem [0:W_DEPTH-1];
    logic [DATA_W-1:0] out_mem [0:O_DEPTH-1];

    // 16-row core instance (unchanged compute core).
    logic clear, in_valid;
    logic [ROWS-1:0] core_row_mask;
    logic [ROWS*DATA_W-1:0] a_rows;
    logic [DATA_W-1:0] b_word;
    logic [COL_W-1:0] in_col;
    logic dump_req, dump_valid;
    logic [ROW_W-1:0] dump_row;
    logic [COL_W-1:0] dump_col;
    logic [ACC_W-1:0] dump_bias, dump_data;
    logic drain_we;
    logic [O_ADDR_W-1:0] drain_waddr;
    logic [DATA_W-1:0] drain_wdata_lo, drain_wdata_hi;
    logic [ACC_W-1:0] drain_bias_comb;

    assign cmd_ready = (state == IDLE);
    assign rsp_valid = (state == RESPONSE);
    assign rsp_status = status_q;

    assign dma_start = (state == LOAD_A) || (state == LOAD_B) ||
                       (state == LOAD_BIAS) || (state == STORE_OUT);
    assign dma_writeback = (state == STORE_OUT);
    assign dma_bank = (state == LOAD_A) ? 2'd0 :
                      (state == LOAD_B) ? 2'd1 : 2'd2;
    assign dma_ext_addr = (state == LOAD_A) ? a_addr_q :
                          (state == LOAD_B) ? b_addr_q :
                          (state == LOAD_BIAS) ? bias_addr_q : out_addr_q;
    assign dma_sram_addr = (state == LOAD_BIAS) ? O_ADDR_W'(BIAS_SRAM) : '0;
    assign dma_words = (state == LOAD_A) ? {5'b0, k_len_q, 4'b0} :
                       (state == LOAD_B) ? 16'd1024 :
                       (state == LOAD_BIAS) ? 16'd32 : 16'd512;

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

    assign a_rdata = '0;
    assign b_rdata = '0;

    assign clear = (state == CLR);
    assign in_valid = (state == RUN);
    assign core_row_mask = row_mask_q;
    assign in_col = run_c_q;

    // The 16 A rows of the current k index are presented in parallel. The
    // A tile is stored as [k][row]; this is a 16-way read of a small array
    // (the main memory-port-structure risk for a real SRAM macro).
    always_comb begin
        for (int r = 0; r < ROWS; r++)
            a_rows[r*DATA_W +: DATA_W] = a_mem[int'(run_k_q)*COLS + r];
        b_word = w_mem[int'(run_k_q)*COLS + int'(run_c_q)];
    end

    kwt_row_broadcast_tile #(
        .ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)
    ) core (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .in_valid(in_valid), .row_mask(core_row_mask),
        .a_rows(a_rows), .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(dump_valid), .dump_data(dump_data)
    );

    // Output capture: bias is read combinationally from the bias SRAM area
    // while the drain request is issued; dump_data is valid one cycle later.
    assign dump_req = (state == DRAIN) && (dreq_idx < 9'd256);
    assign dump_row = ROW_W'(dreq_idx[7:4]);
    assign dump_col = COL_W'(dreq_idx[3:0]);
    assign drain_bias_comb = {
        out_mem[BIAS_SRAM + 2*int'(dreq_idx[3:0]) + 1],
        out_mem[BIAS_SRAM + 2*int'(dreq_idx[3:0])]
    };
    assign dump_bias = drain_bias_comb;
    assign drain_we = (state == DRAIN) && dump_valid;
    assign drain_waddr = O_ADDR_W'(2*int'(cap_idx));
    assign drain_wdata_lo = dump_data[15:0];
    assign drain_wdata_hi = dump_data[31:16];

    // Single write block for the 16-bit SRAM model (DMA ports + drain capture).
    always_ff @(posedge clk) begin
        if (da_en && da_we) a_mem[da_addr] <= da_wdata;
        if (db_en && db_we) w_mem[db_addr] <= db_wdata;
        if (do_en) begin
            if (do_we) out_mem[do_addr] <= do_wdata;
            else out_rdata <= out_mem[do_addr];
        end
        if (drain_we) begin
            out_mem[drain_waddr] <= drain_wdata_lo;
            out_mem[drain_waddr + 1'b1] <= drain_wdata_hi;
        end
    end

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
            status_q <= '0;
            run_k_q <= '0;
            run_c_q <= '0;
            settle_q <= '0;
            dreq_idx <= '0;
            cap_idx <= '0;
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
                    else begin run_k_q <= '0; run_c_q <= '0; state <= CLR; end
                end
                CLR: state <= RUN;
                RUN: begin
                    if (run_c_q == COL_W'(COLS-1)) begin
                        if (run_k_q == k_len_q - 1'b1) begin
                            settle_q <= '0;
                            state <= SETTLE;
                        end else begin
                            run_k_q <= run_k_q + 1'b1;
                            run_c_q <= '0;
                        end
                    end else run_c_q <= run_c_q + 1'b1;
                end
                SETTLE: begin
                    if (settle_q == 4'd7) begin
                        dreq_idx <= '0;
                        cap_idx <= '0;
                        state <= DRAIN;
                    end else settle_q <= settle_q + 1'b1;
                end
                DRAIN: begin
                    if (dreq_idx < 9'd256) dreq_idx <= dreq_idx + 1'b1;
                    if (dump_valid) cap_idx <= cap_idx + 1'b1;
                    if (cap_idx == 9'd256) state <= STORE_OUT;
                end
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
