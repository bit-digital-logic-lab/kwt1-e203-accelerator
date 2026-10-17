module tb_kwt_pingpong_cacc_engine;
    localparam int ROWS = 8;
    localparam int COLS = 16;
    localparam int KLEN = 2;

    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic compute_page = 0;
    logic [6:0] k_len = 7'(KLEN);
    logic [ROWS-1:0] row_mask = '1;
    logic [COLS-1:0] col_mask = '1;
    logic [COLS*32-1:0] bias_data = '0;
    logic relu_en = 0;
    logic load_ready, busy, done, cacc_collision;
    logic load_page = 0;
    logic a_host_en = 0, a_host_we = 0;
    logic b_host_en = 0, b_host_we = 0;
    logic [9:0] a_host_addr = 0, b_host_addr = 0;
    logic [15:0] a_host_wdata = 0, b_host_wdata = 0;
    logic [15:0] a_host_rdata, b_host_rdata;
    logic out_host_en = 0, out_host_we = 0;
    logic [9:0] out_host_addr = 0;
    logic [15:0] out_host_wdata = 0, out_host_rdata;
    int checks = 0;

    always #5 clk = ~clk;

    kwt_pingpong_cacc_engine dut (.*);

    task automatic load_page_data(input bit page);
        int signed aval, bval;
        for (int addr = 0; addr < KLEN*COLS; addr++) begin
            @(negedge clk);
            load_page = page;
            a_host_en = (addr < KLEN*ROWS);
            a_host_we = 1;
            a_host_addr = 10'(addr);
            b_host_en = 1;
            b_host_we = 1;
            b_host_addr = 10'(addr);
            #1;
            if (!load_ready) $fatal(1, "page %0d unexpectedly blocked while loading", page);

            if (page == 0) begin
                int k, r, c;
                k = addr / ROWS;
                r = addr % ROWS;
                aval = (k == 0) ? r+1 : 2*(r+1);
                k = addr / COLS;
                c = addr % COLS;
                bval = c+2+k;
            end else begin
                int k, r, c;
                k = addr / ROWS;
                r = addr % ROWS;
                aval = -(r+1);
                k = addr / COLS;
                c = addr % COLS;
                bval = c+1+k;
            end
            a_host_wdata = 16'(aval);
            b_host_wdata = 16'(bval);
        end
        @(negedge clk);
        a_host_en = 0;
        a_host_we = 0;
        b_host_en = 0;
        b_host_we = 0;
    endtask

    task automatic start_tile(input bit page, input logic [COLS*32-1:0] bias);
        @(negedge clk);
        compute_page = page;
        bias_data = bias;
        start = 1;
        @(negedge clk);
        start = 0;
    endtask

    task automatic wait_done;
        for (int cycle = 0; cycle < 2000; cycle++) begin
            @(negedge clk);
            if (done) begin
                if (busy) $fatal(1, "done and busy overlap");
                if (cacc_collision) $fatal(1, "CACC received multiple products in one row/cycle");
                return;
            end
        end
        $fatal(1, "pingpong CACC engine timeout");
    endtask

    task automatic check_results(input bit page);
        logic [31:0] actual;
        logic signed [31:0] expected;
        int signed total;
        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                @(negedge clk);
                out_host_en = 1;
                out_host_we = 0;
                out_host_addr = 10'((r*COLS+c)*2);
                @(negedge clk);
                actual[15:0] = out_host_rdata;
                out_host_addr = 10'((r*COLS+c)*2+1);
                @(negedge clk);
                actual[31:16] = out_host_rdata;

                if (page == 0) begin
                    total = (r+1)*(3*c+8) + (c-4);
                end else begin
                    total = -((r+1)*(2*c+3)) + 7;
                end
                expected = 32'(total);
                if (actual !== expected)
                    $fatal(1, "page=%0d row=%0d col=%0d got=%0d expected=%0d",
                           page, r, c, $signed(actual), expected);
                checks++;
            end
        end
        @(negedge clk);
        out_host_en = 0;
    endtask

    initial begin
        logic [COLS*32-1:0] bias0, bias1;
        bias0 = '0;
        bias1 = '0;
        for (int c = 0; c < COLS; c++) begin
            bias0[c*32 +: 32] = 32'(c-4);
            bias1[c*32 +: 32] = 32'd7;
        end

        repeat (3) @(negedge clk);
        rst_n = 1;
        load_page_data(0);

        // While page 0 computes, fill page 1 and prove page 0 rejects writes.
        start_tile(0, bias0);
        @(negedge clk);
        load_page = 0;
        a_host_en = 1;
        a_host_we = 1;
        a_host_addr = 0;
        a_host_wdata = 16'h7fff;
        #1;
        if (load_ready) $fatal(1, "active compute page accepted an overwrite");
        @(negedge clk);
        a_host_en = 0;
        a_host_we = 0;
        load_page_data(1);
        wait_done();
        check_results(0);

        // Reuse the other page, clear CACC, and check that no stale sums leak.
        start_tile(1, bias1);
        wait_done();
        check_results(1);

        $display("PASS pingpong+CACC: %0d outputs, inactive-page preload, active-page protection, K accumulation, drain", checks);
        $finish;
    end
endmodule
