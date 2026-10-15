// 脉冲矩阵中的最小计算单元，负责数据传播和一个输出元素的流水累加。
// FORMAT=0 使用有符号整数乘加；FORMAT=1 使用 FP32 乘加。
// FP32 函数参考用户已有的 asr_fp32_alu.sv，次正规数按零处理。
module kwt_pe #(
    parameter int DATA_W = 32,
    parameter int ACC_W = 32,
    parameter int FORMAT = 0
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  acc_clear,

    input  logic [DATA_W-1:0]     a_in,
    input  logic                  a_valid_in,
    input  logic [DATA_W-1:0]     b_in,
    input  logic                  b_valid_in,

    output logic [DATA_W-1:0]     a_out,
    output logic                  a_valid_out,
    output logic [DATA_W-1:0]     b_out,
    output logic                  b_valid_out,

    output logic [ACC_W-1:0]      acc,
    output logic                  acc_update
);
    // 判断 FP32 的特殊编码，并将次正规数转换为带符号的零。
    function automatic logic is_nan(input logic [31:0] x);
        return (&x[30:23]) && (|x[22:0]);
    endfunction

    function automatic logic is_inf(input logic [31:0] x);
        return x[30:0] == 31'h7f800000;
    endfunction

    function automatic logic is_zero(input logic [31:0] x);
        return x[30:0] == 0;
    endfunction

    function automatic logic [31:0] ftz(input logic [31:0] x);
        return (x[30:23] == 0) ? {x[31],31'b0} : x;
    endfunction

    // 对右移时丢弃的低位求或，保留舍入所需的粘滞位。
    function automatic logic [26:0] jam27(
        input logic [26:0] x, input logic [8:0] n
    );
        logic sticky;

        sticky = 1'b0;
        if (n >= 27)
            return {26'b0, |x};

        if (n == 0)
            return x;

        for (int i = 0; i < 27; i++)
            if (i < n) sticky = sticky | x[i];

        return (x >> n) | {26'b0, sticky};
    endfunction

    // 将规格化尾数按最近偶数规则舍入并打包为 FP32。
    function automatic logic [31:0] pack_rne(
        input logic sign, input logic signed [10:0] expn,
        input logic [26:0] sig
    );
        logic [24:0] rounded;
        logic signed [10:0] e;

        e = expn;
        rounded = {1'b0,sig[26:3]} +
                  (sig[2] && (sig[1] || sig[0] || sig[3]));

        if (rounded[24]) begin
            rounded = rounded >> 1;
            e = e + 11'sd1;
        end

        if (sig == 0 || e <= 0)
            return {sign,31'b0};

        if (e >= 255)
            return {sign,8'hff,23'b0};

        return {sign,e[7:0],rounded[22:0]};
    endfunction

    // 数据传播保持单拍，运算流水由下面的格式分支处理。
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            a_valid_out <= !acc_clear && a_valid_in;
            b_valid_out <= !acc_clear && b_valid_in;
        end
    end

    if (FORMAT == 0) begin : g_integer
        localparam int MUL_STAGES = DATA_W / 4;
        localparam int ADD_STAGES = DATA_W / 8;
        localparam int PRODUCT_W = 2 * DATA_W;
        localparam int SEG_W = (ACC_W + ADD_STAGES - 1) / ADD_STAGES;
        localparam int PAD_W = ADD_STAGES * SEG_W;

        logic [DATA_W-1:0] mag_a, mag_b;
        logic [DATA_W-1:0] mul_a_q [0:MUL_STAGES-1];
        logic [DATA_W-1:0] mul_b_q [0:MUL_STAGES-1];
        logic [PRODUCT_W-1:0] mul_partial_q [0:MUL_STAGES-1];
        logic mul_sign_q [0:MUL_STAGES-1];
        logic mul_valid_q [0:MUL_STAGES-1];
        logic signed [PRODUCT_W-1:0] signed_product;

        assign mag_a = a_in[DATA_W-1] ? -a_in : a_in;
        assign mag_b = b_in[DATA_W-1] ? -b_in : b_in;

        // 乘法每级处理乘数的 4 bit，逐级积累移位部分积。
        for (genvar s = 0; s < MUL_STAGES; s++) begin : g_mul
            logic [DATA_W+3:0] digit_product;
            logic [PRODUCT_W-1:0] shifted_product;

            if (s == 0) begin : g_first
                assign digit_product = mag_a * mag_b[3:0];
                assign shifted_product = PRODUCT_W'(digit_product);

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        mul_valid_q[s] <= 1'b0;
                    end else begin
                        mul_valid_q[s] <= !acc_clear && a_valid_in && b_valid_in;
                        mul_a_q[s] <= mag_a;
                        mul_b_q[s] <= mag_b;
                        mul_sign_q[s] <= a_in[DATA_W-1] ^ b_in[DATA_W-1];
                        mul_partial_q[s] <= shifted_product;
                    end
                end
            end else begin : g_next
                assign digit_product = mul_a_q[s-1] * mul_b_q[s-1][4*s +: 4];
                assign shifted_product = PRODUCT_W'(digit_product) << (4*s);

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        mul_valid_q[s] <= 1'b0;
                    end else begin
                        mul_valid_q[s] <= !acc_clear && mul_valid_q[s-1];
                        mul_a_q[s] <= mul_a_q[s-1];
                        mul_b_q[s] <= mul_b_q[s-1];
                        mul_sign_q[s] <= mul_sign_q[s-1];
                        mul_partial_q[s] <= mul_partial_q[s-1] + shifted_product;
                    end
                end
            end
        end

        assign signed_product = mul_sign_q[MUL_STAGES-1] ?
            -$signed(mul_partial_q[MUL_STAGES-1]) :
             $signed(mul_partial_q[MUL_STAGES-1]);

        logic [PAD_W-1:0] add_acc_q [0:ADD_STAGES-1];
        logic [PAD_W-1:0] add_product_q [0:ADD_STAGES-1];
        logic [PAD_W-1:0] add_sum_q [0:ADD_STAGES-1];
        logic add_carry_q [0:ADD_STAGES-1];
        logic add_valid_q [0:ADD_STAGES-1];
        logic [PAD_W-1:0] final_sum;
        logic final_valid;

        // 加法按 ADD_STAGES 段传递进位，最后一级提交部分和。
        for (genvar s = 0; s < ADD_STAGES; s++) begin : g_add
            logic [PAD_W-1:0] operand_acc;
            logic [PAD_W-1:0] operand_product;
            logic [PAD_W-1:0] prior_sum;
            logic prior_carry;
            logic prior_valid;
            logic [SEG_W:0] chunk_sum;
            logic [PAD_W-1:0] merged_sum;

            if (s == 0) begin : g_first
                assign operand_acc = PAD_W'($signed(acc));
                assign operand_product = PAD_W'(signed_product);
                assign prior_sum = '0;
                assign prior_carry = 1'b0;
                assign prior_valid = mul_valid_q[MUL_STAGES-1];
            end else begin : g_next
                assign operand_acc = add_acc_q[s-1];
                assign operand_product = add_product_q[s-1];
                assign prior_sum = add_sum_q[s-1];
                assign prior_carry = add_carry_q[s-1];
                assign prior_valid = add_valid_q[s-1];
            end

            assign chunk_sum = {1'b0,operand_acc[s*SEG_W +: SEG_W]} +
                               {1'b0,operand_product[s*SEG_W +: SEG_W]} +
                               prior_carry;
            assign merged_sum = prior_sum |
                (PAD_W'(chunk_sum[SEG_W-1:0]) << (s*SEG_W));

            if (s == ADD_STAGES-1) begin : g_last
                assign final_sum = merged_sum;
                assign final_valid = prior_valid;
            end else begin : g_pipe
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        add_valid_q[s] <= 1'b0;
                    end else begin
                        add_valid_q[s] <= !acc_clear && prior_valid;
                        add_acc_q[s] <= operand_acc;
                        add_product_q[s] <= operand_product;
                        add_sum_q[s] <= merged_sum;
                        add_carry_q[s] <= chunk_sum[SEG_W];
                    end
                end
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                acc <= '0;
                acc_update <= 1'b0;
            end else if (acc_clear) begin
                acc <= '0;
                acc_update <= 1'b0;
            end else begin
                acc_update <= final_valid;
                if (final_valid)
                    acc <= final_sum[ACC_W-1:0];
            end
        end
    end else begin : g_fp32
        logic [31:0] input_x, input_y;
        logic [23:0] input_mant_a, input_mant_b;
        logic input_special;
        logic [31:0] input_special_value;

        logic [23:0] mul_a_q [0:5];
        logic [23:0] mul_b_q [0:5];
        logic [47:0] mul_partial_q [0:5];
        logic signed [10:0] mul_exp_q [0:5];
        logic mul_sign_q [0:5];
        logic mul_special_q [0:5];
        logic [31:0] mul_special_value_q [0:5];
        logic mul_valid_q [0:5];

        logic [26:0] norm_sig_q;
        logic signed [10:0] norm_exp_q;
        logic norm_sign_q, norm_special_q, norm_valid_q;
        logic [31:0] norm_special_value_q;
        logic [31:0] mul_result_q;
        logic mul_result_valid_q;

        assign input_x = ftz(a_in);
        assign input_y = ftz(b_in);
        assign input_mant_a = {1'b1,input_x[22:0]};
        assign input_mant_b = {1'b1,input_y[22:0]};

        // 第一级处理特殊值和尾数最低 4 bit。
        always_comb begin
            input_special = 1'b1;
            input_special_value = 32'h7fc00000;
            if (is_nan(input_x) || is_nan(input_y) ||
                (is_inf(input_x) && is_zero(input_y)) ||
                (is_inf(input_y) && is_zero(input_x))) begin
                input_special_value = 32'h7fc00000;
            end else if (is_inf(input_x) || is_inf(input_y)) begin
                input_special_value = {input_x[31]^input_y[31],8'hff,23'b0};
            end else if (is_zero(input_x) || is_zero(input_y)) begin
                input_special_value = {input_x[31]^input_y[31],31'b0};
            end else begin
                input_special = 1'b0;
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                mul_valid_q[0] <= 1'b0;
            end else begin
                mul_valid_q[0] <= !acc_clear && a_valid_in && b_valid_in;
                mul_a_q[0] <= input_mant_a;
                mul_b_q[0] <= input_mant_b;
                mul_partial_q[0] <= 48'(input_mant_a * input_mant_b[3:0]);
                mul_exp_q[0] <= $signed({3'b0,input_x[30:23]}) +
                                $signed({3'b0,input_y[30:23]}) - 11'sd127;
                mul_sign_q[0] <= input_x[31] ^ input_y[31];
                mul_special_q[0] <= input_special;
                mul_special_value_q[0] <= input_special_value;
            end
        end

        // 后五级各处理一个 4 bit 尾数片段。
        for (genvar s = 1; s < 6; s++) begin : g_mul_tail
            logic [27:0] digit_product;
            assign digit_product = mul_a_q[s-1] * mul_b_q[s-1][4*s +: 4];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mul_valid_q[s] <= 1'b0;
                end else begin
                    mul_valid_q[s] <= !acc_clear && mul_valid_q[s-1];
                    mul_a_q[s] <= mul_a_q[s-1];
                    mul_b_q[s] <= mul_b_q[s-1];
                    mul_partial_q[s] <= mul_partial_q[s-1] +
                        (48'(digit_product) << (4*s));
                    mul_exp_q[s] <= mul_exp_q[s-1];
                    mul_sign_q[s] <= mul_sign_q[s-1];
                    mul_special_q[s] <= mul_special_q[s-1];
                    mul_special_value_q[s] <= mul_special_value_q[s-1];
                end
            end
        end

        // 第七级完成乘积规格化，第八级完成舍入打包。
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                norm_valid_q <= 1'b0;
                mul_result_valid_q <= 1'b0;
            end else begin
                norm_valid_q <= !acc_clear && mul_valid_q[5];
                mul_result_valid_q <= !acc_clear && norm_valid_q;
                norm_sign_q <= mul_sign_q[5];
                norm_special_q <= mul_special_q[5];
                norm_special_value_q <= mul_special_value_q[5];

                if (mul_partial_q[5][47]) begin
                    norm_sig_q <= mul_partial_q[5][47:21] |
                        {26'b0,(|mul_partial_q[5][20:0])};
                    norm_exp_q <= mul_exp_q[5] + 11'sd1;
                end else begin
                    norm_sig_q <= mul_partial_q[5][46:20] |
                        {26'b0,(|mul_partial_q[5][19:0])};
                    norm_exp_q <= mul_exp_q[5];
                end

                mul_result_q <= norm_special_q ? norm_special_value_q :
                    pack_rne(norm_sign_q,norm_exp_q,norm_sig_q);
            end
        end

        logic [31:0] add0_x_q, add0_y_q;
        logic add0_bypass_q, add0_valid_q;
        logic [31:0] add0_bypass_value_q;
        logic [31:0] add0_x, add0_y;
        logic add0_bypass;
        logic [31:0] add0_bypass_value;

        assign add0_x = ftz(acc);
        assign add0_y = ftz(mul_result_q);

        // 加法第一级处理特殊值，并锁存当前部分和。
        always_comb begin
            add0_bypass = 1'b1;
            add0_bypass_value = 32'h7fc00000;
            if (is_nan(add0_x) || is_nan(add0_y) ||
                (is_inf(add0_x) && is_inf(add0_y) &&
                 add0_x[31] != add0_y[31])) begin
                add0_bypass_value = 32'h7fc00000;
            end else if (is_inf(add0_x)) begin
                add0_bypass_value = add0_x;
            end else if (is_inf(add0_y)) begin
                add0_bypass_value = add0_y;
            end else if (is_zero(add0_x) && is_zero(add0_y)) begin
                add0_bypass_value = {add0_x[31]&add0_y[31],31'b0};
            end else if (is_zero(add0_x)) begin
                add0_bypass_value = add0_y;
            end else if (is_zero(add0_y)) begin
                add0_bypass_value = add0_x;
            end else begin
                add0_bypass = 1'b0;
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                add0_valid_q <= 1'b0;
            end else begin
                add0_valid_q <= !acc_clear && mul_result_valid_q;
                add0_x_q <= add0_x;
                add0_y_q <= add0_y;
                add0_bypass_q <= add0_bypass;
                add0_bypass_value_q <= add0_bypass_value;
            end
        end

        logic [31:0] add1_x, add1_y;
        logic [26:0] add1_mx_q, add1_my_q;
        logic signed [10:0] add1_exp_q;
        logic add1_sign_q, add1_same_q, add1_bypass_q, add1_valid_q;
        logic [31:0] add1_bypass_value_q;

        // 加法第二级交换操作数、对齐指数并生成粘滞位。
        always_comb begin
            add1_x = add0_x_q;
            add1_y = add0_y_q;
            if (add0_x_q[30:0] < add0_y_q[30:0]) begin
                add1_x = add0_y_q;
                add1_y = add0_x_q;
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                add1_valid_q <= 1'b0;
            end else begin
                add1_valid_q <= !acc_clear && add0_valid_q;
                add1_mx_q <= {1'b1,add1_x[22:0],3'b0};
                add1_my_q <= jam27({1'b1,add1_y[22:0],3'b0},
                    {1'b0,add1_x[30:23]} - {1'b0,add1_y[30:23]});
                add1_exp_q <= $signed({3'b0,add1_x[30:23]});
                add1_sign_q <= add1_x[31];
                add1_same_q <= add1_x[31] == add1_y[31];
                add1_bypass_q <= add0_bypass_q;
                add1_bypass_value_q <= add0_bypass_value_q;
            end
        end

        logic [26:0] add2_sig, add2_sig_q;
        logic signed [10:0] add2_exp, add2_exp_q;
        logic add2_sign_q, add2_bypass_q, add2_valid_q;
        logic [31:0] add2_bypass_value_q;
        logic [27:0] add2_wide;
        logic [4:0] add2_shift;

        // 加法第三级计算尾数加减，并规格化异号相减结果。
        always_comb begin
            add2_sig = '0;
            add2_exp = add1_exp_q;
            add2_wide = '0;
            add2_shift = '0;

            if (add1_same_q) begin
                add2_wide = {1'b0,add1_mx_q} + {1'b0,add1_my_q};
                if (add2_wide[27]) begin
                    add2_sig = add2_wide[27:1];
                    add2_sig[0] = add2_sig[0] | add2_wide[0];
                    add2_exp = add1_exp_q + 11'sd1;
                end else begin
                    add2_sig = add2_wide[26:0];
                end
            end else begin
                add2_sig = add1_mx_q - add1_my_q;
                if (add2_sig[26:11] == 0) begin
                    add2_sig = add2_sig << 16;
                    add2_shift[4] = 1'b1;
                end
                if (add2_sig[26:19] == 0) begin
                    add2_sig = add2_sig << 8;
                    add2_shift[3] = 1'b1;
                end
                if (add2_sig[26:23] == 0) begin
                    add2_sig = add2_sig << 4;
                    add2_shift[2] = 1'b1;
                end
                if (add2_sig[26:25] == 0) begin
                    add2_sig = add2_sig << 2;
                    add2_shift[1] = 1'b1;
                end
                if (!add2_sig[26]) begin
                    add2_sig = add2_sig << 1;
                    add2_shift[0] = 1'b1;
                end
                add2_exp = add1_exp_q - $signed({6'b0,add2_shift});
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                add2_valid_q <= 1'b0;
            end else begin
                add2_valid_q <= !acc_clear && add1_valid_q;
                add2_sig_q <= add2_sig;
                add2_exp_q <= add2_exp;
                add2_sign_q <= add1_sign_q;
                add2_bypass_q <= add1_bypass_q;
                add2_bypass_value_q <= add1_bypass_value_q;
            end
        end

        // 加法第四级完成舍入，提交结果并产生完成脉冲。
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                acc <= '0;
                acc_update <= 1'b0;
            end else if (acc_clear) begin
                acc <= '0;
                acc_update <= 1'b0;
            end else begin
                acc_update <= add2_valid_q;
                if (add2_valid_q) begin
                    if (add2_bypass_q)
                        acc <= add2_bypass_value_q;
                    else if (add2_sig_q == 0)
                        acc <= 32'b0;
                    else
                        acc <= pack_rne(add2_sign_q,add2_exp_q,add2_sig_q);
                end
            end
        end
    end
endmodule
