// Two-page A/B operand storage. The compute side reads its selected page;
// the load side can fill the other page at the same time.
module kwt_operand_pingpong #(
    parameter int DATA_W = 16,
    parameter int DEPTH = 1024,
    parameter int ADDR_W = (DEPTH <= 2) ? 1 : $clog2(DEPTH)
) (
    input logic clk, rst_n,
    input logic load_page,
    input logic load_ready,
    input logic a_host_en, a_host_we,
    input logic [ADDR_W-1:0] a_host_addr,
    input logic [DATA_W-1:0] a_host_wdata,
    input logic b_host_en, b_host_we,
    input logic [ADDR_W-1:0] b_host_addr,
    input logic [DATA_W-1:0] b_host_wdata,
    input logic compute_en,
    input logic compute_page,
    input logic [ADDR_W-1:0] compute_a_addr,
    input logic [ADDR_W-1:0] compute_b_addr,
    output logic [DATA_W-1:0] compute_a_data,
    output logic [DATA_W-1:0] compute_b_data
);
    logic [DATA_W-1:0] a_mem [0:1][0:DEPTH-1];
    logic [DATA_W-1:0] b_mem [0:1][0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (a_host_en && a_host_we && load_ready)
            a_mem[load_page][a_host_addr] <= a_host_wdata;
        if (b_host_en && b_host_we && load_ready)
            b_mem[load_page][b_host_addr] <= b_host_wdata;
        if (compute_en) begin
            compute_a_data <= a_mem[compute_page][compute_a_addr];
            compute_b_data <= b_mem[compute_page][compute_b_addr];
        end
    end
endmodule
