// Self-contained Verilator differential/boundary testbench for the KWT-1
// FFN1 tile engine (`kwt_pingpong_cacc_engine`, 8x16 lanes, INT16 A/W,
// INT32 accumulator/bias/output).
//
// This bench is deliberately independent from the Python reference model:
// it only loads an externally supplied stimulus and dumps the raw 32-bit
// outputs. The reference (expected) values are produced by the driver
// `run_extended_verilator.py` from the frozen FFN1 weights and an integer
// GEMM, never from the DUT output.
//
// One simulation invocation runs exactly one `+CASES`-selected case:
//   +CASE=<name> +A=<file> +B=<file> +BIAS=<file> +OUT=<file>
//   +KLEN=<int> +ROWMASK=<hex> +COLMASK=<hex> +RELU=<0|1>
//   +MODE=normal|reset|pingpong   [+OUT2=<file> for pingpong]
//
// A layout in `+A`: word index k*ROWS + r  (r = row within tile)
// B layout in `+B`: word index k*COLS + c  (c = column within tile)
// BIAS: 32-bit words per output column, COLS words.
// OUT/OUT2: 128 lines of 8-hex-digit INT32, row-major r*COLS+c.

module tb_kwt_extended_regression;
    localparam int ROWS = 8;
    localparam int COLS = 16;
    localparam int ACC_W = 32;
    localparam int TILE_ELEMS = ROWS * COLS;

    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic compute_page = 0;
    logic [6:0] k_len = 0;
    logic [ROWS-1:0] row_mask = 0;
    logic [COLS-1:0] col_mask = 0;
    logic [COLS*ACC_W-1:0] bias_data = 0;
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
    string case_name = "case";
    string a_path, b_path, bias_path, out_path, out2_path, mode = "normal";

    always #5 clk = ~clk;

    kwt_pingpong_cacc_engine dut (.*);

    // ---------------------------------------------------------------- helpers
    task automatic load_operands(input int klen, input bit page);
        int fa, fb, code;
        logic [15:0] av, bv;
        fa = $fopen(a_path, "r");
        fb = $fopen(b_path, "r");
        if (fa == 0 || fb == 0) $fatal(1, "cannot open A/B stimulus");
        for (int addr = 0; addr < klen * COLS; addr++) begin
            @(negedge clk);
            load_page = page;
            if (addr < klen * ROWS) begin
                code = $fscanf(fa, "%h", av);
                a_host_en = 1; a_host_we = 1;
                a_host_addr = 10'(addr); a_host_wdata = av;
            end else begin
                a_host_en = 0; a_host_we = 0;
            end
            code = $fscanf(fb, "%h", bv);
            b_host_en = 1; b_host_we = 1;
            b_host_addr = 10'(addr); b_host_wdata = bv;
            #1;
            if (!load_ready) $fatal(1, "idle page rejected a load at addr %0d", addr);
        end
        @(negedge clk);
        a_host_en = 0; a_host_we = 0; b_host_en = 0; b_host_we = 0;
        $fclose(fa); $fclose(fb);
    endtask

    task automatic load_bias;
        int fd, code;
        logic [31:0] word;
        fd = $fopen(bias_path, "r");
        if (fd == 0) $fatal(1, "cannot open bias stimulus");
        bias_data = '0;
        for (int c = 0; c < COLS; c++) begin
            code = $fscanf(fd, "%h", word);
            bias_data[c*ACC_W +: ACC_W] = word;
        end
        $fclose(fd);
    endtask

    task automatic fire_start(input bit page);
        @(negedge clk);
        compute_page = page;
        start = 1;
        @(negedge clk);
        start = 0;
    endtask

    task automatic wait_done;
        for (int cycle = 0; cycle < 4000; cycle++) begin
            @(negedge clk);
            if (done) begin
                if (busy) $fatal(1, "done and busy asserted together");
                if (cacc_collision) $fatal(1, "CACC row/cycle collision");
                return;
            end
        end
        $fatal(1, "tile engine timeout");
    endtask

    task automatic dump_results(input string path);
        int fd;
        logic [31:0] actual;
        fd = $fopen(path, "w");
        if (fd == 0) $fatal(1, "cannot open output file %s", path);
        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                @(negedge clk);
                out_host_en = 1; out_host_we = 0;
                out_host_addr = 10'((r*COLS + c) * 2);
                @(negedge clk);
                actual[15:0] = out_host_rdata;
                out_host_addr = 10'((r*COLS + c) * 2 + 1);
                @(negedge clk);
                actual[31:16] = out_host_rdata;
                $fwrite(fd, "%08x\n", actual);
                checks++;
            end
        end
        @(negedge clk);
        out_host_en = 0;
        $fclose(fd);
    endtask

    task automatic run_normal;
        load_operands(k_len, 1'b0);
        load_bias;
        fire_start(1'b0);
        wait_done;
        dump_results(out_path);
    endtask

    // Reset injected in the middle of a running tile; the engine must not
    // report completion, and a clean re-run must still match the model.
    task automatic run_reset;
        load_operands(k_len, 1'b0);
        load_bias;
        fire_start(1'b0);
        repeat (12) @(negedge clk);
        if (done) $fatal(1, "reset case finished before reset injection");
        rst_n = 0;
        repeat (4) @(negedge clk);
        if (done) $fatal(1, "done asserted while in reset");
        if (busy) $fatal(1, "busy asserted while in reset");
        rst_n = 1;
        repeat (2) @(negedge clk);
        load_operands(k_len, 1'b0);
        load_bias;
        fire_start(1'b0);
        wait_done;
        dump_results(out_path);
    endtask

    // Back-pressure / ping-pong handshake: while page 0 computes, a write to
    // the active page must be refused, the idle page must accept a preload,
    // and the preloaded page must reproduce the same result.
    task automatic run_pingpong;
        load_operands(k_len, 1'b0);
        load_bias;
        fire_start(1'b0);
        @(negedge clk);
        load_page = 0;
        a_host_en = 1; a_host_we = 1; a_host_addr = 0; a_host_wdata = 16'hdead;
        #1;
        if (load_ready) $fatal(1, "active compute page accepted an overwrite");
        @(negedge clk);
        a_host_en = 0; a_host_we = 0;
        load_operands(k_len, 1'b1);
        wait_done;
        dump_results(out_path);
        repeat (2) @(negedge clk);
        fire_start(1'b1);
        wait_done;
        dump_results(out2_path);
    endtask

    // ------------------------------------------------------------------- main
    initial begin
        logic [31:0] rmask, cmask;
        int klen, relu;
        if (!$value$plusargs("A=%s", a_path)) $fatal(1, "missing +A");
        if (!$value$plusargs("B=%s", b_path)) $fatal(1, "missing +B");
        if (!$value$plusargs("BIAS=%s", bias_path)) $fatal(1, "missing +BIAS");
        if (!$value$plusargs("OUT=%s", out_path)) $fatal(1, "missing +OUT");
        if (!$value$plusargs("KLEN=%d", klen)) $fatal(1, "missing +KLEN");
        if (!$value$plusargs("ROWMASK=%h", rmask)) $fatal(1, "missing +ROWMASK");
        if (!$value$plusargs("COLMASK=%h", cmask)) $fatal(1, "missing +COLMASK");
        void'($value$plusargs("CASE=%s", case_name));
        if (!$value$plusargs("RELU=%d", relu)) relu = 0;
        void'($value$plusargs("MODE=%s", mode));
        void'($value$plusargs("OUT2=%s", out2_path));

        k_len = 7'(klen);
        row_mask = rmask[ROWS-1:0];
        col_mask = cmask[COLS-1:0];
        relu_en = relu[0];

        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        if (mode == "reset") run_reset;
        else if (mode == "pingpong") run_pingpong;
        else run_normal;

        $display("CASE_DONE %s mode=%s checks=%0d", case_name, mode, checks);
        $finish;
    end
endmodule
