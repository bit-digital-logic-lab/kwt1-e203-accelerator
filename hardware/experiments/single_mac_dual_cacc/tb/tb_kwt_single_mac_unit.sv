// Focused RTL microtest for kwt_single_mac_dual_cacc. No task/bus model.
//
// Proves, against an independent in-SV INT32 reference:
//   * reset clears both pages and deasserts valid/ready,
//   * ONE multiplier array feeds page 0 and page 1 with page ownership,
//   * the MAC feed->CACC latency (a product is not dumpable for 1 cycle),
//   * feed/export busy conflicts drive in_ready / dump_ready low,
//   * export is a streaming 1/cycle port with request gaps (backpressure),
//   * clear[] invalidates exactly one page; invalid cells export bias only,
//   * a mid-run reset invalidates everything,
//   * dump order and data (bias added at export only) are exact.
module tb_kwt_single_mac_unit;
    localparam int ROWS = 4, COLS = 16, K = 2, M = 2*ROWS;
    localparam int DATA_W = 16, ACC_W = 32;
    localparam int COL_W = $clog2(COLS);
    localparam int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS);
    localparam int PAGE_W = 1;

    logic clk = 0, rst_n = 0;
    logic in_valid = 0;
    logic in_ready;
    logic [PAGE_W-1:0] in_page = 0;
    logic [ROWS-1:0] row_mask = '0;
    logic [ROWS*DATA_W-1:0] a_rows = '0;
    logic [DATA_W-1:0] b_word = '0;
    logic [3:0] in_col = '0;
    logic [1:0] clear = '0;
    logic dump_req = 0;
    logic dump_ready;
    logic [PAGE_W-1:0] dump_page = 0;
    logic [ROW_W-1:0] dump_row = '0;
    logic [3:0] dump_col = '0;
    logic [ACC_W-1:0] dump_bias = '0;
    logic dump_valid;
    logic [PAGE_W-1:0] dump_valid_page;
    logic [ROW_W-1:0] dump_valid_row;
    logic [3:0] dump_valid_col;
    logic [PAGE_W+ROW_W+COL_W-1:0] dump_addr;
    logic [ACC_W-1:0] dump_data;

    logic [15:0] a_mem [0:M-1][0:K-1];
    logic [15:0] w_mem [0:K-1][0:COLS-1];
    logic [31:0] bias_mem [0:COLS-1];
    int errors = 0;

    always #5 clk = ~clk;
    initial begin
        #200000;
        $fatal(1, "single-mac unit watchdog expired");
    end

    kwt_single_mac_dual_cacc #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready), .in_page(in_page),
        .row_mask(row_mask), .a_rows(a_rows), .b_word(b_word), .in_col(in_col),
        .clear(clear),
        .dump_req(dump_req), .dump_ready(dump_ready), .dump_page(dump_page),
        .dump_row(dump_row), .dump_col(dump_col), .dump_bias(dump_bias),
        .dump_valid(dump_valid), .dump_valid_page(dump_valid_page),
        .dump_valid_row(dump_valid_row), .dump_valid_col(dump_valid_col),
        .dump_addr(dump_addr), .dump_data(dump_data)
    );

    function automatic logic [31:0] reference(input int row, input int col);
        logic signed [31:0] acc;
        acc = $signed(bias_mem[col]);
        for (int k = 0; k < K; k++)
            acc = acc + $signed(a_mem[row][k]) * $signed(w_mem[k][col]);
        return acc;
    endfunction

    task automatic fail(input string msg);
        $display("FAIL: %s", msg);
        errors++;
    endtask

    task automatic drive_feed(input int page, input int col, input int k, input int row0);
        in_valid = 1'b1; in_page = PAGE_W'(page); in_col = 4'(col);
        b_word = w_mem[k][col];
        for (int r = 0; r < ROWS; r++)
            a_rows[r*DATA_W +: DATA_W] = a_mem[row0+r][k];
    endtask

    // Drive one feed beat and hold until accepted (valid/ready).
    task automatic feed_beat(input int page, input int col, input int k, input int row0);
        @(negedge clk);
        drive_feed(page, col, k, row0);
        #1;
        while (!in_ready) begin @(negedge clk); #1; end
    endtask

    task automatic dump_cell(input int page, input int row, input int col,
                             input int gap, input bit check, input bit single = 1'b0);
        logic [31:0] exp;
        if (single)
            exp = $signed(bias_mem[col]) +
                  $signed(a_mem[page*ROWS+row][0]) * $signed(w_mem[0][col]);
        else
            exp = reference(page*ROWS + row, col);
        repeat (gap) begin
            @(negedge clk);
            dump_req = 1'b0;
        end
        @(negedge clk);
        dump_req = 1'b1; dump_page = PAGE_W'(page);
        dump_row = ROW_W'(row); dump_col = 4'(col); dump_bias = bias_mem[col];
        #1;
        while (!dump_ready) begin @(negedge clk); #1; end
        @(negedge clk);
        dump_req = 1'b0;
        #1;
        if (check) begin
            if (dump_valid !== 1'b1)
                fail($sformatf("dump_valid low at page%0d (%0d,%0d)", page, row, col));
            else if (dump_data !== exp)
                fail($sformatf("page%0d (%0d,%0d) got=%08h exp=%08h", page, row, col,
                               dump_data, exp));
            if (dump_valid_page !== PAGE_W'(page) || dump_valid_row !== ROW_W'(row) ||
                dump_valid_col !== 4'(col))
                fail($sformatf("dump_addr mismatch at page%0d (%0d,%0d)", page, row, col));
            if (dump_addr !== {PAGE_W'(page), ROW_W'(row), 4'(col)})
                fail($sformatf("dump_addr bus mismatch at page%0d (%0d,%0d)", page, row, col));
        end
    endtask

    task automatic clear_page(input int page);
        @(negedge clk);
        clear = (page == 0) ? 2'b01 : 2'b10;
        @(negedge clk);
        clear = 2'b00;
    endtask

    initial begin : run
        for (int r = 0; r < M; r++)
            for (int k = 0; k < K; k++)
                a_mem[r][k] = 16'(((r*3 + k*5) % 9) - 4);
        for (int k = 0; k < K; k++)
            for (int c = 0; c < COLS; c++)
                w_mem[k][c] = 16'(((k*7 + c*2) % 7) - 3);
        for (int c = 0; c < COLS; c++)
            bias_mem[c] = 32'((c % 5) - 2);
        row_mask = {ROWS{1'b1}};

        // ---- 1. reset: both pages invalid, no valid/ready ----
        repeat (4) @(negedge clk);
        if (dump_valid !== 1'b0) fail("dump_valid high after reset");
        if (in_ready !== 1'b0) fail("in_ready high during reset");
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        #1;
        if (in_ready !== 1'b1) fail("in_ready low after reset with no conflict");

        // ---- 2. feed tile0 (rows 0..3) into page 0 ----
        for (int k = 0; k < K; k++)
            for (int c = 0; c < COLS; c++)
                feed_beat(0, c, k, 0);

        // ---- 3. MAC latency: product of the last beat is not dumpable yet ----
        @(negedge clk);
        in_valid = 1'b0;
        #1;
        if (dump_ready !== 1'b0) fail("dump_ready high while product is in flight");
        @(negedge clk); #1;
        if (dump_ready !== 1'b1) fail("dump_ready still low after accumulate");

        // ---- 4. export page 0 with request gaps; verify ordering + data ----
        for (int i = 0; i < ROWS*COLS; i++)
            dump_cell(0, i/COLS, i%COLS, (i % 11 == 5) ? 1 : 0, 1'b1);

        // ---- 5. real overlap: feed page 1 while streaming page 0 out ----
        begin
            int feed_idx, dump_idx;
            feed_idx = 0; dump_idx = 0;
            @(negedge clk);
            while (feed_idx < K*COLS || dump_idx < ROWS*COLS) begin
                in_valid = 1'b0; dump_req = 1'b0;
                if (feed_idx < K*COLS) begin
                    drive_feed(1, feed_idx % COLS, feed_idx/COLS, ROWS);
                    feed_idx++;
                end
                if (dump_idx < ROWS*COLS) begin
                    dump_req = 1'b1; dump_page = 1'b0;
                    dump_row = ROW_W'(dump_idx/COLS); dump_col = 4'(dump_idx%COLS);
                    dump_bias = bias_mem[dump_idx%COLS];
                end
                #1;
                @(negedge clk);   // exactly one cycle: each beat/dump issued once
                #1;
                if (dump_idx < ROWS*COLS) begin
                    if (dump_valid !== 1'b1)
                        fail($sformatf("overlap dump_valid low at idx %0d", dump_idx));
                    else if (dump_data !== reference(dump_idx/COLS, dump_idx%COLS))
                        fail($sformatf("overlap dump mismatch at idx %0d", dump_idx));
                    dump_idx++;
                end
            end
            @(negedge clk);
            in_valid = 1'b0; dump_req = 1'b0;
        end

        // ---- 6. export page 1 (fed during the page-0 export above) ----
        for (int i = 0; i < ROWS*COLS; i++)
            dump_cell(1, i/COLS, i%COLS, 0, 1'b1);

        // ---- 7. dedicated busy-conflict checks (after page checks) ----
        // Accept one page-0 dump request, then in the following cycle test that
        // feeding the exporting page is blocked but the other page is free.
        @(negedge clk);
        dump_req = 1'b1; dump_page = 1'b0; dump_row = '0; dump_col = '0;
        dump_bias = bias_mem[0]; #1;
        if (dump_ready !== 1'b1) fail("page0 dump_ready low when idle");
        @(negedge clk);
        dump_req = 1'b0;
        #1;
        if (dump_valid !== 1'b1) fail("page0 baseline dump_valid low");
        in_valid = 1'b1; in_page = 1'b0; in_col = '0; #1;
        if (in_ready !== 1'b0) fail("in_ready high for page under export");
        in_page = 1'b1; #1;
        if (in_ready !== 1'b1) fail("in_ready low for free page during export");
        in_valid = 1'b0; in_page = 1'b0; #1;
        // feed (page 1) and dump (page 1) in the same cycle: feed wins, dump waits.
        @(negedge clk);
        drive_feed(1, 0, 0, ROWS);
        dump_req = 1'b1; dump_page = 1'b1; dump_row = '0; dump_col = '0;
        dump_bias = bias_mem[0]; #1;
        if (dump_ready !== 1'b0) fail("dump_ready high for page being fed");
        if (in_ready !== 1'b1) fail("in_ready low while feeding free page");
        @(negedge clk);
        in_valid = 1'b0; dump_req = 1'b0; #1;

        // ---- 8. clear page 1, feed one beat, invalid cells export bias only ----
        clear_page(1);
        feed_beat(1, 0, 0, ROWS);
        @(negedge clk);
        in_valid = 1'b0;
        dump_cell(1, 0, 0, 0, 1'b1, 1'b1);            // one beat only (k=0)
        // Column 3 was never written after the clear: must export bias only.
        @(negedge clk);
        dump_req = 1'b1; dump_page = 1'b1; dump_row = '0; dump_col = 4'd3;
        dump_bias = bias_mem[3]; #1;
        while (!dump_ready) begin @(negedge clk); #1; end
        @(negedge clk); #1;
        if (dump_valid !== 1'b1) fail("invalid-cell dump_valid low");
        else if (dump_data !== bias_mem[3])
            fail($sformatf("cleared invalid cell got=%08h exp=bias=%08h",
                           dump_data, bias_mem[3]));
        @(negedge clk);
        dump_req = 1'b0;

        // ---- 9. mid-run reset invalidates both pages ----
        @(negedge clk);
        rst_n = 1'b0; in_valid = 1'b0; dump_req = 1'b0;
        repeat (2) @(negedge clk);
        #1;
        if (dump_valid !== 1'b0) fail("dump_valid survived mid-run reset");
        if (in_ready !== 1'b0) fail("in_ready high during mid-run reset");
        rst_n = 1'b1;
        row_mask = {ROWS{1'b1}};
        @(negedge clk);
        feed_beat(0, 0, 0, 0);
        @(negedge clk);
        in_valid = 1'b0;
        @(negedge clk);
        dump_cell(0, 0, 0, 0, 1'b1, 1'b1);   // single k=0 beat after reset

        if (errors != 0) $fatal(1, "single-mac unit FAILED with %0d errors", errors);
        $display("PASS single-mac unit: reset, ownership, latency, backpressure, clear");
        $finish;
    end
endmodule
