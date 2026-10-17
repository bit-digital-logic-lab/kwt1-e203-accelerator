// 单块计算测试：检查 SRAM 装载、两轮 K 累加、写回、偏置和 ReLU。
module tb_kwt_tile_engine;
    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic [6:0] k_len = 2;
    logic [7:0] row_mask = '1;
    logic [15:0] col_mask = '1;
    logic relu_en = 0;
    logic [511:0] bias_data = '0;
    logic busy, done;
    logic a_host_en = 0, a_host_we = 0;
    logic b_host_en = 0, b_host_we = 0;
    logic out_host_en = 0;
    logic out_host_we = 0;
    logic [9:0] a_host_addr = 0, b_host_addr = 0, out_host_addr = 0;
    logic [15:0] a_host_wdata = 0, b_host_wdata = 0;
    logic [15:0] out_host_wdata = 0;
    logic [15:0] a_host_rdata, b_host_rdata, out_host_rdata;
    int checks = 0;

    // 时钟及完整控制器实例。
    always #5 clk = ~clk;
    kwt_tile_engine dut (.*);

    // 通过主机端口写入两个 K 切片。
    task automatic load_inputs;
        for (int k = 0; k < 2; k++) begin
            for (int r = 0; r < 8; r++) begin
                @(negedge clk);
                a_host_en = 1;
                a_host_we = 1;
                a_host_addr = 10'(k*8+r);
                a_host_wdata = 16'(k == 0 ? r+1 : -(r+1));
            end
        end
        @(negedge clk);
        a_host_en = 0;
        a_host_we = 0;

        for (int k = 0; k < 2; k++) begin
            for (int c = 0; c < 16; c++) begin
                @(negedge clk);
                b_host_en = 1;
                b_host_we = 1;
                b_host_addr = 10'(k*16+c);
                b_host_wdata = 16'(c+1+k);
            end
        end
        @(negedge clk);
        b_host_en = 0;
        b_host_we = 0;
    endtask

    // 填满两块输入 SRAM，验证最大 K=64 的地址边界。
    task automatic load_full_inputs;
        for (int addr = 0; addr < 512; addr++) begin
            @(negedge clk);
            a_host_en = 1;
            a_host_we = 1;
            a_host_addr = 10'(addr);
            a_host_wdata = 16'd1;
        end
        @(negedge clk);
        a_host_en = 0;
        a_host_we = 0;

        for (int addr = 0; addr < 1024; addr++) begin
            @(negedge clk);
            b_host_en = 1;
            b_host_we = 1;
            b_host_addr = 10'(addr);
            b_host_wdata = 16'd1;
        end
        @(negedge clk);
        b_host_en = 0;
        b_host_we = 0;
    endtask

    // 等待完成脉冲，并保证超时能直接报告故障。
    task automatic run_tile;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        for (int cycle = 0; cycle < 5000; cycle++) begin
            if (done) begin
                if (busy) $fatal(1, "done and busy overlap");
                @(negedge clk);
                return;
            end
            @(negedge clk);
        end
        $fatal(1, "tile engine timeout");
    endtask

    // 同步读回 256 个半字，逐项检查整个输出块。
    task automatic check_output(input bit use_relu, input bit full_k);
        logic [31:0] actual;
        logic [31:0] expected;
        int signed total;

        for (int r = 0; r < 8; r++) begin
            for (int c = 0; c < 16; c++) begin
                @(negedge clk);
                out_host_en = 1;
                out_host_addr = 10'((r*16+c)*2);
                @(negedge clk);
                actual[15:0] = out_host_rdata;
                out_host_addr = 10'((r*16+c)*2+1);
                @(negedge clk);
                actual[31:16] = out_host_rdata;

                total = full_k ? 64 : -(r+1);
                if (use_relu) total += c-3;
                if (use_relu && total < 0) total = 0;
                expected = (row_mask[r] && col_mask[c]) ? 32'(total) : 0;
                if (actual !== expected)
                    $fatal(1, "output mismatch mode=%0d row=%0d col=%0d got=%h expected=%h",
                           use_relu, r, c, actual, expected);
                checks++;
            end
        end
        @(negedge clk);
        out_host_en = 0;
    endtask

    // 第一轮验证纯矩阵累加；第二轮验证边界屏蔽与后处理。
    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        load_inputs();
        run_tile();
        check_output(0, 0);

        row_mask[7] = 0;
        col_mask[15] = 0;
        relu_en = 1;
        for (int c = 0; c < 16; c++)
            bias_data[c*32 +: 32] = 32'(c-3);
        run_tile();
        check_output(1, 0);

        row_mask = '1;
        col_mask = '1;
        relu_en = 0;
        bias_data = '0;
        k_len = 64;
        load_full_inputs();
        run_tile();
        check_output(0, 1);

        $display("PASS tile engine: %0d output checks", checks);
        $finish;
    end
endmodule
