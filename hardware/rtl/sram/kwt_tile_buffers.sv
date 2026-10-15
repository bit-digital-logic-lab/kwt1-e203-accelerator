// 三块独立的 16 bit × 1024 缓存，分别存放 A、B 和输出数据。
// 当前为单口同步 SRAM 行为模型，后续需替换为赛题提供的存储宏。
module kwt_tile_buffers (
    input  logic        clk,

    input  logic        a_en,
    input  logic        a_we,
    input  logic [9:0]  a_addr,
    input  logic [15:0] a_wdata,
    output logic [15:0] a_rdata,

    input  logic        b_en,
    input  logic        b_we,
    input  logic [9:0]  b_addr,
    input  logic [15:0] b_wdata,
    output logic [15:0] b_rdata,

    input  logic        out_en,
    input  logic        out_we,
    input  logic [9:0]  out_addr,
    input  logic [15:0] out_wdata,
    output logic [15:0] out_rdata
);
    // 三个缓存拥有各自的读写端口，互不占用访问周期。
    logic [15:0] a_mem [0:1023];
    logic [15:0] b_mem [0:1023];
    logic [15:0] out_mem [0:1023];

    // 在时钟上升沿访问缓存；读数据在上升沿后更新。
    always_ff @(posedge clk) begin
        // A 缓存的读写控制。
        if (a_en) begin
            if (a_we)
                a_mem[a_addr] <= a_wdata;
            else
                a_rdata <= a_mem[a_addr];
        end

        // B 缓存的读写控制。
        if (b_en) begin
            if (b_we)
                b_mem[b_addr] <= b_wdata;
            else
                b_rdata <= b_mem[b_addr];
        end

        // 输出缓存的读写控制。
        if (out_en) begin
            if (out_we)
                out_mem[out_addr] <= out_wdata;
            else
                out_rdata <= out_mem[out_addr];
        end
    end
endmodule
