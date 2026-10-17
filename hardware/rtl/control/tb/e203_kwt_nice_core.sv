`include "e203_defines.v"

`ifndef KWT_E203_EXT_MEM_BYTES
`define KWT_E203_EXT_MEM_BYTES 2048
`endif

// Test-only replacement for the open E203 example NICE co-unit.
// The upstream CPU RTL is compiled without its example NICE module.
module e203_subsys_nice_core (
    input  logic nice_clk, nice_rst_n,
    output logic nice_active, nice_mem_holdup,
    input  logic nice_req_valid,
    output logic nice_req_ready,
    input  logic [`E203_XLEN-1:0] nice_req_inst,
    input  logic [`E203_XLEN-1:0] nice_req_rs1, nice_req_rs2,
    output logic nice_rsp_valid,
    input  logic nice_rsp_ready,
    output logic [`E203_XLEN-1:0] nice_rsp_rdat,
    output logic nice_rsp_err,
    output logic nice_icb_cmd_valid,
    input  logic nice_icb_cmd_ready,
    output logic [`E203_ADDR_SIZE-1:0] nice_icb_cmd_addr,
    output logic nice_icb_cmd_read,
    output logic [`E203_XLEN-1:0] nice_icb_cmd_wdata,
    output logic [1:0] nice_icb_cmd_size,
    input  logic nice_icb_rsp_valid,
    output logic nice_icb_rsp_ready,
    input  logic [`E203_XLEN-1:0] nice_icb_rsp_rdata,
    input  logic nice_icb_rsp_err
);
    wire bus_valid, bus_write, bus_ready, bus_rvalid;
    wire [31:0] bus_addr;
    wire [127:0] bus_wdata, bus_rdata;
    wire [15:0] bus_wstrb;

    assign nice_active = nice_req_valid || !nice_req_ready;
    assign nice_mem_holdup = 1'b0;
    assign nice_icb_cmd_valid = 1'b0;
    assign nice_icb_cmd_addr = '0;
    assign nice_icb_cmd_read = 1'b1;
    assign nice_icb_cmd_wdata = '0;
    assign nice_icb_cmd_size = 2'b10;
    assign nice_icb_rsp_ready = 1'b1;

    kwt_nice_system_model #(.EXT_MEM_BYTES(`KWT_E203_EXT_MEM_BYTES)) accelerator (
        .clk(nice_clk), .rst_n(nice_rst_n),
        .nice_req_valid(nice_req_valid), .nice_req_ready(nice_req_ready),
        .nice_req_instr(nice_req_inst), .nice_req_rs1(nice_req_rs1),
        .nice_req_rs2(nice_req_rs2),
        .nice_rsp_valid(nice_rsp_valid), .nice_rsp_ready(nice_rsp_ready),
        .nice_rsp_data(nice_rsp_rdat), .nice_rsp_err(nice_rsp_err),
        .bus_valid(bus_valid), .bus_write(bus_write), .bus_ready(bus_ready),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_wstrb(bus_wstrb),
        .bus_rvalid(bus_rvalid), .bus_rdata(bus_rdata)
    );

    kwt_ext_memory_model #(.MEM_BYTES(`KWT_E203_EXT_MEM_BYTES)) shared_ram (
        .clk(nice_clk), .rst_n(nice_rst_n),
        .valid(bus_valid), .write(bus_write), .ready(bus_ready),
        .byte_addr(bus_addr), .wdata(bus_wdata), .wstrb(bus_wstrb),
        .rvalid(bus_rvalid), .rdata(bus_rdata)
    );
endmodule
