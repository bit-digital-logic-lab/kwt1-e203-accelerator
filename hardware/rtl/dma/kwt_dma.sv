`include "kwt_memory_defs.svh"

// One descriptor moves consecutive 16-bit words. External byte address is
// 16-byte aligned; SRAM address is a word address. Only bank 2 can write back.
module kwt_dma #(
    parameter int EXT_MEM_BYTES = `KWT_EXT_MEM_BYTES,
    parameter int SRAM_DATA_W = `KWT_SRAM_DATA_W,
    parameter int SRAM_DEPTH = `KWT_SRAM_DEPTH,
    parameter int BUS_W = `KWT_DMA_BUS_W,
    parameter int SRAM_AW = $clog2(SRAM_DEPTH)
) (
    input  logic clk, rst_n,
    input  logic start,
    input  logic writeback,
    input  logic [1:0] bank,
    input  logic [31:0] ext_byte_addr,
    input  logic [SRAM_AW-1:0] sram_word_addr,
    input  logic [15:0] word_count,
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
    localparam int LANES = BUS_W / SRAM_DATA_W;
    localparam int LANE_W = $clog2(LANES);
    localparam int BUS_BYTES = BUS_W / 8;

    typedef enum logic [2:0] {
        IDLE, LOAD_REQ, LOAD_WAIT, LOAD_FILL,
        STORE_READ, STORE_CAPTURE, STORE_REQ, FINISH
    } state_t;
    state_t state;
    logic [1:0] active_bank;
    logic [31:0] active_ext_addr;
    logic [SRAM_AW-1:0] active_sram_addr;
    logic [15:0] remaining;
    logic [LANE_W-1:0] lane;
    logic [BUS_W-1:0] packed_data;
    logic [BUS_BYTES-1:0] packed_strb;
    logic [SRAM_DATA_W-1:0] selected_rdata;
    logic sram_en, sram_we;
    logic [SRAM_DATA_W-1:0] sram_wdata;
    logic [BUS_W-1:0] captured_data;

    assign busy = state != IDLE && state != FINISH;
    assign done = state == FINISH;
    assign bus_valid = state == LOAD_REQ || state == STORE_REQ;
    assign bus_write = state == STORE_REQ;
    assign bus_addr = active_ext_addr;
    assign bus_wdata = packed_data;
    assign bus_wstrb = packed_strb;
    // During writeback, capture the previous synchronous SRAM read while
    // issuing the next address. One SRAM read port is sufficient.
    assign sram_en = state == LOAD_FILL || state == STORE_READ ||
                     (state == STORE_CAPTURE && remaining > 1 &&
                      lane != LANE_W'(LANES-1));
    assign sram_we = state == LOAD_FILL;
    assign sram_wdata = captured_data[lane*SRAM_DATA_W +: SRAM_DATA_W];
    assign a_en = sram_en && active_bank == 0;
    assign b_en = sram_en && active_bank == 1;
    assign out_en = sram_en && active_bank == 2;
    assign a_we = a_en && sram_we;
    assign b_we = b_en && sram_we;
    assign out_we = out_en && sram_we;
    assign a_addr = active_sram_addr;
    assign b_addr = active_sram_addr;
    assign out_addr = active_sram_addr;
    assign a_wdata = sram_wdata;
    assign b_wdata = sram_wdata;
    assign out_wdata = sram_wdata;
    assign selected_rdata = active_bank == 0 ? a_rdata :
                            active_bank == 1 ? b_rdata : out_rdata;

    initial begin
        if (SRAM_DATA_W < 8 || SRAM_DATA_W % 8 != 0 ||
            BUS_W <= SRAM_DATA_W || BUS_W % SRAM_DATA_W != 0 ||
            (LANES & (LANES - 1)) != 0 ||
            SRAM_DEPTH < 2 || SRAM_DEPTH > 65535 ||
            (SRAM_DEPTH & (SRAM_DEPTH - 1)) != 0 ||
            EXT_MEM_BYTES < BUS_BYTES || EXT_MEM_BYTES % BUS_BYTES != 0)
            $fatal(1, "kwt_dma: unsupported memory parameters");
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            error <= 0;
            active_bank <= 0;
            active_ext_addr <= 0;
            active_sram_addr <= 0;
            remaining <= 0;
            lane <= 0;
            packed_data <= 0;
            packed_strb <= 0;
            captured_data <= 0;
        end else begin
            case (state)
                IDLE: if (start) begin
                    error <= 0;
                    // Reserve a full bus beat even for a partial final beat.
                    if (bank == 3 || (writeback && bank != 2) ||
                        ext_byte_addr % BUS_BYTES != 0 ||
                        word_count == 0 ||
                        (32'(sram_word_addr) + 32'(word_count)) > SRAM_DEPTH ||
                        (64'(ext_byte_addr) +
                         ((64'(word_count) + 64'(LANES) - 1) / 64'(LANES)) *
                         64'(BUS_BYTES)) > 64'(EXT_MEM_BYTES)) begin
                        error <= 1;
                        state <= FINISH;
                    end else begin
                        active_bank <= bank;
                        active_ext_addr <= ext_byte_addr;
                        active_sram_addr <= sram_word_addr;
                        remaining <= word_count;
                        lane <= 0;
                        packed_data <= 0;
                        packed_strb <= 0;
                        if (writeback) state <= STORE_READ;
                        else state <= LOAD_REQ;
                    end
                end
                LOAD_REQ: if (bus_ready) state <= LOAD_WAIT;
                LOAD_WAIT: if (bus_rvalid) begin
                    captured_data <= bus_rdata;
                    lane <= 0;
                    state <= LOAD_FILL;
                end
                LOAD_FILL: begin
                    remaining <= remaining - 1'b1;
                    active_sram_addr <= active_sram_addr + 1'b1;
                    if (remaining == 1) state <= FINISH;
                    else if (lane == LANE_W'(LANES-1)) begin
                        active_ext_addr <= active_ext_addr + BUS_BYTES;
                        state <= LOAD_REQ;
                    end else lane <= lane + 1'b1;
                end
                STORE_READ: begin
                    // The read uses today's address; advance for the next
                    // read while its data is being registered by the SRAM.
                    active_sram_addr <= active_sram_addr + 1'b1;
                    state <= STORE_CAPTURE;
                end
                STORE_CAPTURE: begin
                    packed_data[lane*SRAM_DATA_W +: SRAM_DATA_W] <= selected_rdata;
                    packed_strb[lane*(SRAM_DATA_W/8) +: SRAM_DATA_W/8] <= '1;
                    remaining <= remaining - 1'b1;
                    if (remaining > 1 && lane != LANE_W'(LANES-1))
                        active_sram_addr <= active_sram_addr + 1'b1;
                    if (remaining == 1 || lane == LANE_W'(LANES-1)) state <= STORE_REQ;
                    else begin
                        lane <= lane + 1'b1;
                        state <= STORE_CAPTURE;
                    end
                end
                STORE_REQ: if (bus_ready) begin
                    if (remaining == 0) state <= FINISH;
                    else begin
                        active_ext_addr <= active_ext_addr + BUS_BYTES;
                        packed_data <= 0;
                        packed_strb <= 0;
                        lane <= 0;
                        state <= STORE_READ;
                    end
                end
                FINISH: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
