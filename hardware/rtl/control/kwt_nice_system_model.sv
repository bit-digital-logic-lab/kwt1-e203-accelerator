// Provisional integration: one NICE-like submit command, a two-beat descriptor
// fetch, the tile controller, DMA, and a shared 128-bit simulation bus.
// This is not the competition SoC's System Bus or full E203 NICE adapter.
module kwt_nice_system_model #(
    parameter int EXT_MEM_BYTES = 512 * 1024
) (
    input  logic clk, rst_n,
    input  logic nice_req_valid,
    output logic nice_req_ready,
    input  logic [31:0] nice_req_instr,
    input  logic [31:0] nice_req_rs1,
    input  logic [31:0] nice_req_rs2,
    output logic nice_rsp_valid,
    input  logic nice_rsp_ready,
    output logic [31:0] nice_rsp_data,
    output logic nice_rsp_err,

    output logic bus_valid, bus_write,
    input  logic bus_ready,
    output logic [31:0] bus_addr,
    output logic [127:0] bus_wdata,
    output logic [15:0] bus_wstrb,
    input  logic bus_rvalid,
    input  logic [127:0] bus_rdata
);
    // Custom-0, funct3=110: xd=1, xs1=1, xs2=0; funct7=0.
    localparam logic [6:0] OPCODE_CUSTOM0 = 7'b0001011;
    typedef enum logic [2:0] {
        IDLE, FETCH0_REQ, FETCH0_WAIT, FETCH1_REQ, FETCH1_WAIT,
        ISSUE_TILE, WAIT_TILE, RESPONSE
    } state_t;
    state_t state;

    logic [31:0] desc_addr;
    logic [31:0] a_addr_q, b_addr_q, bias_addr_q, out_addr_q;
    logic [6:0] k_len_q;
    logic [7:0] row_mask_q;
    logic [15:0] col_mask_q;
    logic relu_q;
    logic [1:0] status_q;
    logic tile_cmd_ready, tile_rsp_valid;
    logic [1:0] tile_rsp_status;
    logic dma_bus_valid, dma_bus_write;
    logic [31:0] dma_bus_addr;
    logic [127:0] dma_bus_wdata;
    logic [15:0] dma_bus_wstrb;
    logic desc_bus_owner;

    assign nice_req_ready = state == IDLE;
    assign nice_rsp_valid = state == RESPONSE;
    assign nice_rsp_data = {30'b0, status_q};
    assign nice_rsp_err = status_q != 0;
    assign desc_bus_owner = state == FETCH0_REQ || state == FETCH0_WAIT ||
                            state == FETCH1_REQ || state == FETCH1_WAIT;
    assign bus_valid = desc_bus_owner ?
                       (state == FETCH0_REQ || state == FETCH1_REQ) : dma_bus_valid;
    assign bus_write = desc_bus_owner ? 1'b0 : dma_bus_write;
    assign bus_addr = desc_bus_owner ?
                      desc_addr + ((state == FETCH1_REQ || state == FETCH1_WAIT) ? 32'd16 : 32'd0) :
                      dma_bus_addr;
    assign bus_wdata = desc_bus_owner ? 128'b0 : dma_bus_wdata;
    assign bus_wstrb = desc_bus_owner ? 16'b0 : dma_bus_wstrb;

    kwt_tile_system_model #(.EXT_MEM_BYTES(EXT_MEM_BYTES)) tile_system (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(state == ISSUE_TILE), .cmd_ready(tile_cmd_ready),
        .cmd_a_addr(a_addr_q), .cmd_b_addr(b_addr_q),
        .cmd_bias_addr(bias_addr_q), .cmd_out_addr(out_addr_q),
        .cmd_k_len(k_len_q), .cmd_row_mask(row_mask_q),
        .cmd_col_mask(col_mask_q), .cmd_relu_en(relu_q),
        .rsp_valid(tile_rsp_valid), .rsp_ready(state == WAIT_TILE),
        .rsp_status(tile_rsp_status),
        .bus_valid(dma_bus_valid), .bus_write(dma_bus_write),
        .bus_ready(bus_ready), .bus_addr(dma_bus_addr),
        .bus_wdata(dma_bus_wdata), .bus_wstrb(dma_bus_wstrb),
        .bus_rvalid(bus_rvalid && !desc_bus_owner), .bus_rdata(bus_rdata)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            desc_addr <= '0;
            a_addr_q <= '0;
            b_addr_q <= '0;
            bias_addr_q <= '0;
            out_addr_q <= '0;
            k_len_q <= '0;
            row_mask_q <= '0;
            col_mask_q <= '0;
            relu_q <= 1'b0;
            status_q <= '0;
        end else begin
            case (state)
                IDLE: if (nice_req_valid) begin
                    status_q <= 0;
                    desc_addr <= nice_req_rs1;
                    if (nice_req_instr[6:0] != OPCODE_CUSTOM0 ||
                        nice_req_instr[14:12] != 3'b110 ||
                        nice_req_instr[31:25] != 0 ||
                        |nice_req_rs1[3:0] ||
                        (64'(nice_req_rs1) + 64'd32) > 64'(EXT_MEM_BYTES)) begin
                        status_q <= 2'd3; // invalid instruction or descriptor address
                        state <= RESPONSE;
                    end else state <= FETCH0_REQ;
                end
                FETCH0_REQ: if (bus_ready) state <= FETCH0_WAIT;
                FETCH0_WAIT: if (bus_rvalid) begin
                    a_addr_q <= bus_rdata[31:0];
                    b_addr_q <= bus_rdata[63:32];
                    bias_addr_q <= bus_rdata[95:64];
                    out_addr_q <= bus_rdata[127:96];
                    state <= FETCH1_REQ;
                end
                FETCH1_REQ: if (bus_ready) state <= FETCH1_WAIT;
                FETCH1_WAIT: if (bus_rvalid) begin
                    k_len_q <= bus_rdata[6:0];
                    relu_q <= bus_rdata[7];
                    row_mask_q <= bus_rdata[39:32];
                    col_mask_q <= bus_rdata[55:40];
                    state <= ISSUE_TILE;
                end
                ISSUE_TILE: if (tile_cmd_ready) state <= WAIT_TILE;
                WAIT_TILE: if (tile_rsp_valid) begin
                    status_q <= tile_rsp_status;
                    state <= RESPONSE;
                end
                RESPONSE: if (nice_rsp_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
