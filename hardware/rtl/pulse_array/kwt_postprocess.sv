// 矩阵结果的整数后处理：可加偏置，并选择直通或 ReLU。
// GELU 所需的定点缩放尚未定义，本模块不将 ReLU 当作 GELU 使用。
module kwt_postprocess (
    input  logic [31:0] acc_data,
    input  logic [31:0] bias_data,
    input  logic        relu_en,
    output logic [31:0] result
);
    // 使用 32 bit 回绕加法，与整数 PE 的累加宽度一致。
    logic [31:0] biased;
    assign biased = acc_data + bias_data;

    // ReLU 仅用于明确选择该模式的任务。
    assign result = (relu_en && biased[31]) ? 32'b0 : biased;
endmodule
