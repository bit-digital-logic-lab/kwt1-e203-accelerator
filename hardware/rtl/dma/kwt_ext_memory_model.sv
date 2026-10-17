`include "kwt_memory_defs.svh"

// Simulation model of a 128-bit, byte-write-enabled external memory.
module kwt_ext_memory_model #(
    parameter int MEM_BYTES = `KWT_EXT_MEM_BYTES,
    parameter int BUS_W = `KWT_DMA_BUS_W
) (
    input  logic clk, rst_n,
    input  logic valid, write,
    output logic ready,
    input  logic [31:0] byte_addr,
    input  logic [BUS_W-1:0] wdata,
    input  logic [BUS_W/8-1:0] wstrb,
    output logic rvalid,
    output logic [BUS_W-1:0] rdata
);
    localparam int BUS_BYTES = BUS_W / 8;
    logic [BUS_W-1:0] mem [0:MEM_BYTES/BUS_BYTES-1];
    assign ready = 1'b1;
    initial begin
        if (MEM_BYTES % BUS_BYTES != 0)
            $fatal(1, "kwt_ext_memory_model: memory size must align to bus");
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rvalid <= 0;
            rdata <= '0;
        end else begin
            rvalid <= valid && ready && !write;
            if (valid && ready && byte_addr < MEM_BYTES &&
                byte_addr % BUS_BYTES == 0) begin
                if (write) begin
                    for (int i = 0; i < BUS_BYTES; i++)
                        if (wstrb[i])
                            mem[byte_addr/BUS_BYTES][i*8 +: 8] <= wdata[i*8 +: 8];
                end else rdata <= mem[byte_addr/BUS_BYTES];
            end
        end
    end
endmodule
