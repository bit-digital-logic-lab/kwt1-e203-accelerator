`include "kwt_memory_defs.svh"

// Row-major tile transfer over the existing one-row DMA. Strides describe
// the distance between row starts; each row still uses aligned bus beats.
module kwt_dma_2d #(
    parameter int EXT_MEM_BYTES = `KWT_EXT_MEM_BYTES,
    parameter int SRAM_DATA_W = `KWT_SRAM_DATA_W,
    parameter int SRAM_DEPTH = `KWT_SRAM_DEPTH,
    parameter int BUS_W = `KWT_DMA_BUS_W,
    parameter int SRAM_AW = $clog2(SRAM_DEPTH)
) (
    input  logic clk, rst_n,
    input  logic start, writeback,
    input  logic [1:0] bank,
    input  logic [31:0] ext_byte_addr,
    input  logic [SRAM_AW-1:0] sram_word_addr,
    input  logic [15:0] rows, words_per_row,
    input  logic [31:0] ext_stride_bytes,
    input  logic [15:0] sram_stride_words,
    output logic busy, done, error,

    output logic bus_valid, bus_write,
    input  logic bus_ready,
    output logic [31:0] bus_addr,
    output logic [BUS_W-1:0] bus_wdata,
    output logic [BUS_W/8-1:0] bus_wstrb,
    input  logic bus_rvalid,
    input  logic [BUS_W-1:0] bus_rdata,

    output logic a_en, a_we,
    output logic [SRAM_AW-1:0] a_addr,
    output logic [SRAM_DATA_W-1:0] a_wdata,
    input  logic [SRAM_DATA_W-1:0] a_rdata,
    output logic b_en, b_we,
    output logic [SRAM_AW-1:0] b_addr,
    output logic [SRAM_DATA_W-1:0] b_wdata,
    input  logic [SRAM_DATA_W-1:0] b_rdata,
    output logic out_en, out_we,
    output logic [SRAM_AW-1:0] out_addr,
    output logic [SRAM_DATA_W-1:0] out_wdata,
    input  logic [SRAM_DATA_W-1:0] out_rdata
);
    localparam int BUS_BYTES = BUS_W / 8;
    localparam int LANES = BUS_W / SRAM_DATA_W;
    typedef enum logic [2:0] {IDLE, CHECK, LAUNCH, WAIT_ROW, FINISH} state_t;
    state_t state;
    logic [15:0] row_index;
    logic [15:0] active_rows, active_words;
    logic [15:0] check_rows_left;
    logic [63:0] check_ext_addr, check_sram_addr;
    logic [15:0] active_sram_stride;
    logic [31:0] active_ext_stride;
    logic [1:0] active_bank;
    logic active_writeback;
    logic [31:0] row_ext_addr;
    logic [SRAM_AW-1:0] row_sram_addr;
    logic inner_done, inner_error;

    assign busy = state == CHECK || state == LAUNCH || state == WAIT_ROW;
    assign done = state == FINISH;

    kwt_dma #(
        .EXT_MEM_BYTES(EXT_MEM_BYTES), .SRAM_DATA_W(SRAM_DATA_W),
        .SRAM_DEPTH(SRAM_DEPTH), .BUS_W(BUS_W), .SRAM_AW(SRAM_AW)
    ) row_dma (
        .clk, .rst_n, .start(state == LAUNCH),
        .writeback(active_writeback), .bank(active_bank),
        .ext_byte_addr(row_ext_addr), .sram_word_addr(row_sram_addr),
        .word_count(active_words), .busy(), .done(inner_done),
        .error(inner_error),
        .bus_valid, .bus_write, .bus_ready, .bus_addr, .bus_wdata,
        .bus_wstrb, .bus_rvalid, .bus_rdata,
        .a_en, .a_we, .a_addr, .a_wdata, .a_rdata,
        .b_en, .b_we, .b_addr, .b_wdata, .b_rdata,
        .out_en, .out_we, .out_addr, .out_wdata, .out_rdata
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            error <= 0;
            row_index <= 0;
            active_rows <= 0;
            active_words <= 0;
            check_rows_left <= 0;
            check_ext_addr <= 0;
            check_sram_addr <= 0;
            active_sram_stride <= 0;
            active_ext_stride <= 0;
            active_bank <= 0;
            active_writeback <= 0;
            row_ext_addr <= 0;
            row_sram_addr <= 0;
        end else begin
            case (state)
                IDLE: if (start) begin
                    error <= 0;
                    // Check all row bounds before any bus or SRAM side effect.
                    // A short sequential preflight avoids a stride multiplier.
                    if (rows == 0 || words_per_row == 0 ||
                        bank == 3 || (writeback && bank != 2) ||
                        ext_byte_addr % BUS_BYTES != 0 ||
                        ext_stride_bytes % BUS_BYTES != 0) begin
                        error <= 1;
                        state <= FINISH;
                    end else begin
                        row_index <= 0;
                        active_rows <= rows;
                        active_words <= words_per_row;
                        active_sram_stride <= sram_stride_words;
                        active_ext_stride <= ext_stride_bytes;
                        active_bank <= bank;
                        active_writeback <= writeback;
                        check_rows_left <= rows;
                        check_ext_addr <= 64'(ext_byte_addr);
                        check_sram_addr <= 64'(sram_word_addr);
                        row_ext_addr <= ext_byte_addr;
                        row_sram_addr <= sram_word_addr;
                        state <= CHECK;
                    end
                end
                CHECK: begin
                    if (check_ext_addr +
                        ((64'(active_words) + LANES - 1) / LANES) * BUS_BYTES
                            > EXT_MEM_BYTES ||
                        check_sram_addr + 64'(active_words) > SRAM_DEPTH) begin
                        error <= 1;
                        state <= FINISH;
                    end else if (check_rows_left == 1) begin
                        state <= LAUNCH;
                    end else begin
                        check_rows_left <= check_rows_left - 1'b1;
                        check_ext_addr <= check_ext_addr + 64'(active_ext_stride);
                        check_sram_addr <= check_sram_addr + 64'(active_sram_stride);
                    end
                end
                LAUNCH: state <= WAIT_ROW;
                WAIT_ROW: if (inner_done) begin
                    if (inner_error) begin
                        error <= 1;
                        state <= FINISH;
                    end else if (row_index == active_rows - 1'b1) begin
                        state <= FINISH;
                    end else begin
                        row_index <= row_index + 1'b1;
                        row_ext_addr <= row_ext_addr + active_ext_stride;
                        row_sram_addr <= row_sram_addr + SRAM_AW'(active_sram_stride);
                        state <= LAUNCH;
                    end
                end
                FINISH: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
