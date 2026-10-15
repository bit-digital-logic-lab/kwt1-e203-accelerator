// 脉冲矩阵测试台：检查阵列传播、累加、掩码、清零和复位。
module tb_kwt_pulse_array #(
    parameter int DATA_W = 32,
    parameter int ACC_W = 32,
    parameter int FORMAT = 0
);
    localparam int ROWS = 8;
    localparam int COLS = 16;
    localparam int MUL_CYCLES = (FORMAT == 1) ? 8 : DATA_W / 4;
    localparam int ADD_CYCLES = (FORMAT == 1) ? 4 : DATA_W / 8;
    localparam int MAC_CYCLES = MUL_CYCLES + ADD_CYCLES;

    logic clk;
    logic rst_n = 0;
    logic acc_clear = 0;
    logic launch = 0;
    logic [ROWS*DATA_W-1:0] a_data = '0;
    logic [ROWS-1:0] a_mask = '0;
    logic [COLS*DATA_W-1:0] b_data = '0;
    logic [COLS-1:0] b_mask = '0;
    logic [ROWS*COLS*ACC_W-1:0] acc_data;
    logic [ROWS*COLS-1:0] acc_update;
    int checks = 0;
    int timing_checks = 0;

    // 单独产生时钟，避免与测试流程中的控制信号混在一起。
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // 被测模块始终采用完整的 8×16 阵列规模。
    kwt_pulse_array #(
        .ROWS(ROWS), .COLS(COLS),
        .DATA_W(DATA_W), .ACC_W(ACC_W), .FORMAT(FORMAT)
    ) dut (
        .clk(clk), .rst_n(rst_n), .acc_clear(acc_clear),
        .launch(launch), .a_data(a_data), .a_mask(a_mask),
        .b_data(b_data), .b_mask(b_mask),
        .acc_data(acc_data), .acc_update(acc_update)
    );

    // 生成在 FP32 中精确可表示的 2 的整数次幂。
    function automatic logic [31:0] fp32_pow2(input int exponent, input bit sign);
        logic [7:0] biased;
        biased = 8'(exponent + 127);
        return {sign,biased,23'b0};
    endfunction

    // 整数输入取不同的行列值，便于发现传播接线错误。
    function automatic int integer_a(input int row, input int wave);
        case (wave)
            0: return row + 1;
            1: return -(row + 2);
            default: return 2 * row + 1;
        endcase
    endfunction

    function automatic int integer_b(input int col, input int wave);
        case (wave)
            0: return col + 1;
            1: return col + 2;
            default: return -(col + 1);
        endcase
    endfunction

    // 将一个 K 序号的左右边界向量装入输入端口。
    task automatic prepare_wave(input int wave, input bit mask_edges);
        a_data = '0;
        b_data = '0;
        a_mask = '1;
        b_mask = '1;

        if (mask_edges) begin
            a_mask[ROWS-1] = 1'b0;
            b_mask[COLS-1] = 1'b0;
        end

        for (int r = 0; r < ROWS; r++) begin
            if (FORMAT == 1)
                a_data[r*DATA_W +: DATA_W] = fp32_pow2(r, 1'b0);
            else
                a_data[r*DATA_W +: DATA_W] = DATA_W'(integer_a(r,wave));
        end

        for (int c = 0; c < COLS; c++) begin
            if (FORMAT == 1)
                b_data[c*DATA_W +: DATA_W] =
                    fp32_pow2(-c - ((wave == 2) ? 1 : 0), wave == 2);
            else
                b_data[c*DATA_W +: DATA_W] = DATA_W'(integer_b(c,wave));
        end
    endtask

    // 发射一次脉冲，逐拍检查完成信号与位置和流水深度一致。
    task automatic issue_wave(input int wave, input bit mask_edges);
        bit expected_update;

        @(negedge clk);
        prepare_wave(wave, mask_edges);
        launch = 1'b1;

        for (int step = 0; step < ROWS + COLS + MAC_CYCLES + 1; step++) begin
            @(negedge clk);
            launch = 1'b0;

            for (int r = 0; r < ROWS; r++) begin
                for (int c = 0; c < COLS; c++) begin
                    expected_update = a_mask[r] && b_mask[c] &&
                        (step == r + c + MAC_CYCLES - 1);

                    if (acc_update[r*COLS+c] !== expected_update)
                        $fatal(1, "TIMING FAIL format=%0d width=%0d wave=%0d step=%0d PE=(%0d,%0d) got=%b expected=%b",
                               FORMAT, DATA_W, wave, step, r, c,
                               acc_update[r*COLS+c], expected_update);

                    timing_checks++;
                end
            end
        end
    endtask

    // 按当前波数计算每个 PE 应有的结果。
    function automatic logic [ACC_W-1:0] expected_value(
        input int row, input int col, input int wave_count
    );
        bit active_second;
        longint signed total;

        active_second = (row != ROWS-1) && (col != COLS-1);

        if (FORMAT == 1) begin
            if (wave_count == 1 || (wave_count == 2 && !active_second))
                return ACC_W'(fp32_pow2(row-col, 1'b0));

            if (wave_count == 2)
                return ACC_W'(fp32_pow2(row-col+1, 1'b0));

            if (active_second)
                return ACC_W'({1'b0,8'(127+row-col),23'h400000});

            return ACC_W'(fp32_pow2(row-col-1, 1'b0));
        end

        total = integer_a(row,0) * integer_b(col,0);
        if (wave_count >= 2 && active_second)
            total += integer_a(row,1) * integer_b(col,1);
        if (wave_count >= 3)
            total += integer_a(row,2) * integer_b(col,2);

        return ACC_W'(total);
    endfunction

    // 比较全部 128 个 PE 的部分和，错误立即终止仿真。
    task automatic check_array(input int wave_count);
        logic [ACC_W-1:0] actual;
        logic [ACC_W-1:0] expected;

        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                actual = acc_data[(r*COLS+c)*ACC_W +: ACC_W];
                expected = (wave_count == 0) ? '0 : expected_value(r,c,wave_count);

                if (actual !== expected)
                    $fatal(1, "FAIL format=%0d width=%0d wave=%0d PE=(%0d,%0d) got=%h expected=%h",
                           FORMAT, DATA_W, wave_count, r, c, actual, expected);

                checks++;
            end
        end
    endtask

    // 以 MAC_CYCLES 为最小间隔连续发射两项，检查累加反馈。
    task automatic check_min_interval;
        logic [ACC_W-1:0] actual;
        logic [ACC_W-1:0] expected;
        longint signed integer_sum;

        @(negedge clk);
        prepare_wave(0, 1'b0);
        launch = 1'b1;

        @(negedge clk);
        launch = 1'b0;
        repeat (MAC_CYCLES-1) @(negedge clk);
        prepare_wave(1, 1'b0);
        launch = 1'b1;

        @(negedge clk);
        launch = 1'b0;
        repeat (ROWS + COLS + MAC_CYCLES + 1) @(negedge clk);

        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                actual = acc_data[(r*COLS+c)*ACC_W +: ACC_W];

                if (FORMAT == 1) begin
                    expected = ACC_W'(fp32_pow2(r-c+1, 1'b0));
                end else begin
                    integer_sum = integer_a(r,0) * integer_b(c,0) +
                                  integer_a(r,1) * integer_b(c,1);
                    expected = ACC_W'(integer_sum);
                end

                if (actual !== expected)
                    $fatal(1, "FEEDBACK FAIL format=%0d width=%0d PE=(%0d,%0d) got=%h expected=%h",
                           FORMAT, DATA_W, r, c, actual, expected);

                checks++;
            end
        end
    endtask

    // 顺序执行复位、三轮数据、掩码和累加器清零检查。
    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        check_array(0);

        issue_wave(0, 1'b0);
        check_array(1);

        issue_wave(1, 1'b1);
        check_array(2);

        issue_wave(2, 1'b0);
        check_array(3);

        @(negedge clk);
        acc_clear = 1'b1;
        @(negedge clk);
        acc_clear = 1'b0;
        check_array(0);

        issue_wave(0, 1'b0);
        check_array(1);

        @(negedge clk);
        acc_clear = 1'b1;
        @(negedge clk);
        acc_clear = 1'b0;
        check_min_interval();

        rst_n = 1'b0;
        repeat (2) @(negedge clk);
        check_array(0);

        $display("PASS format=%0d data_w=%0d acc_w=%0d mul=%0d add=%0d result_checks=%0d timing_checks=%0d",
                 FORMAT, DATA_W, ACC_W, MUL_CYCLES, ADD_CYCLES,
                 checks, timing_checks);
        $finish;
    end
endmodule
