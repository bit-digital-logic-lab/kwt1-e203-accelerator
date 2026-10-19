// Focused RTL microtest for the dual-CACC scheduling primitive used by
// tb_kwt_streaming_task: two UNCHANGED kwt_row_broadcast_tile cores, export of
// page A overlapping accumulation into page B.
//
// It proves, without any task/bus model:
//   * reset clears both accumulator pages (dump_valid stays 0),
//   * a tile can be exported (dump) in the same cycles the other page
//     accumulates, with dump_req gaps (back-pressure) that must not corrupt
//     ordering or data,
//   * export values equal an independent INT32 reference computed in SV,
//   * a mid-run reset invalidates both pages (no stale accumulator).
//
// The full task testbench additionally checks against an external numpy INT32
// reference over tiny / ragged / overflow / two-tile / full FFN1 shapes.
module tb_kwt_dualacc_unit;
    localparam int ROWS = 4, COLS = 16, K = 3, M = 2*ROWS;
    localparam int DATA_W = 16;
    localparam int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS);

    logic clk = 0, rst_n = 0;
    logic [ROWS*DATA_W-1:0] a_rows0 = '0, a_rows1 = '0;
    logic [15:0] b_word0 = '0, b_word1 = '0;
    logic [3:0] in_col0 = '0, in_col1 = '0;
    logic [ROWS-1:0] row_mask0 = '0, row_mask1 = '0;
    logic in_valid0 = 0, in_valid1 = 0, clear0 = 0, clear1 = 0;
    logic dump_req0 = 0, dump_req1 = 0;
    logic [ROW_W-1:0] dump_row0 = '0, dump_row1 = '0;
    logic [3:0] dump_col0 = '0, dump_col1 = '0;
    logic [31:0] dump_bias0 = '0, dump_bias1 = '0;
    logic dump_valid0, dump_valid1;
    logic [31:0] dump_data0, dump_data1;

    logic [15:0] a_mem [0:M-1][0:K-1];
    logic [15:0] w_mem [0:K-1][0:COLS-1];
    logic signed [31:0] bias_mem [0:COLS-1];

    int  sched_valid [0:1023];
    int  sched_col   [0:1023];
    int  sched_k     [0:1023];
    int  sched_len = 0;
    int  errors = 0;

    always #5 clk = ~clk;
    initial begin
        #200000;
        $fatal(1, "dualacc unit watchdog expired");
    end

    kwt_row_broadcast_tile #(.ROWS(ROWS), .DATA_W(DATA_W)) page0 (
        .clk(clk), .rst_n(rst_n), .clear(clear0),
        .in_valid(in_valid0), .row_mask(row_mask0), .a_rows(a_rows0),
        .b_word(b_word0), .in_col(in_col0),
        .dump_req(dump_req0), .dump_row(dump_row0), .dump_col(dump_col0),
        .dump_bias(dump_bias0), .dump_valid(dump_valid0), .dump_data(dump_data0)
    );
    kwt_row_broadcast_tile #(.ROWS(ROWS), .DATA_W(DATA_W)) page1 (
        .clk(clk), .rst_n(rst_n), .clear(clear1),
        .in_valid(in_valid1), .row_mask(row_mask1), .a_rows(a_rows1),
        .b_word(b_word1), .in_col(in_col1),
        .dump_req(dump_req1), .dump_row(dump_row1), .dump_col(dump_col1),
        .dump_bias(dump_bias1), .dump_valid(dump_valid1), .dump_data(dump_data1)
    );

    function automatic int reference(input int row, input int col);
        int acc;
        acc = int'(bias_mem[col]);
        for (int k = 0; k < K; k++)
            acc += int'($signed(a_mem[row][k])) * int'($signed(w_mem[k][col]));
        return acc;
    endfunction

    function automatic int cell_k(input int idx); return idx / COLS; endfunction
    function automatic int cell_c(input int idx); return idx % COLS; endfunction

    task automatic build_sched(input bit with_gaps);
        sched_len = 0;
        for (int k = 0; k < K; k++)
            for (int c = 0; c < COLS; c++) begin
                if (with_gaps && (k*COLS+c) % 7 == 0) begin
                    sched_valid[sched_len] = 0; sched_col[sched_len] = 0;
                    sched_k[sched_len] = 0;     sched_len++;
                end
                sched_valid[sched_len] = 1; sched_col[sched_len] = c;
                sched_k[sched_len] = k;     sched_len++;
            end
    endtask

    task automatic stream_step(input int core, input int idx, input int row0);
        if (idx >= sched_len || !sched_valid[idx]) return;
        if (core == 0) begin
            in_valid0 = 1'b1; in_col0 = 4'(sched_col[idx]);
            b_word0 = w_mem[sched_k[idx]][sched_col[idx]];
            for (int r = 0; r < ROWS; r++)
                a_rows0[r*DATA_W +: DATA_W] = a_mem[row0+r][sched_k[idx]];
        end else begin
            in_valid1 = 1'b1; in_col1 = 4'(sched_col[idx]);
            b_word1 = w_mem[sched_k[idx]][sched_col[idx]];
            for (int r = 0; r < ROWS; r++)
                a_rows1[r*DATA_W +: DATA_W] = a_mem[row0+r][sched_k[idx]];
        end
    endtask

    task automatic issue_dump(input int core, input int idx);
        int rr, cc;
        rr = cell_k(idx); cc = cell_c(idx);
        if (core == 0) begin
            dump_req0 = 1'b1; dump_row0 = ROW_W'(rr); dump_col0 = 4'(cc);
            dump_bias0 = bias_mem[cc];
        end else begin
            dump_req1 = 1'b1; dump_row1 = ROW_W'(rr); dump_col1 = 4'(cc);
            dump_bias1 = bias_mem[cc];
        end
    endtask

    task automatic check_cell(input int core, input int row0, input int idx);
        int rr, cc, gexp;
        logic [31:0] got;
        rr = cell_k(idx); cc = cell_c(idx);
        gexp = reference(row0+rr, cc);
        if (core == 0) begin
            got = dump_data0;
            if (dump_valid0 !== 1'b1) begin
                $error("page0 dump_valid low at (%0d,%0d)", rr, cc); errors++;
            end else if (got !== gexp) begin
                $error("page0 (%0d,%0d) got=%0d exp=%0d", rr, cc, got, gexp); errors++;
            end
        end else begin
            got = dump_data1;
            if (dump_valid1 !== 1'b1) begin
                $error("page1 dump_valid low at (%0d,%0d)", rr, cc); errors++;
            end else if (got !== gexp) begin
                $error("page1 (%0d,%0d) got=%0d exp=%0d", rr, cc, got, gexp); errors++;
            end
        end
    endtask

    task automatic stream_only(input int core, input int row0, input bit with_gaps);
        build_sched(with_gaps);
        for (int i = 0; i < sched_len; i++) begin
            @(negedge clk);
            in_valid0 = 0; in_valid1 = 0;
            stream_step(core, i, row0);
        end
        @(negedge clk);
        in_valid0 = 0; in_valid1 = 0;
    endtask

    // Export cells of one tile, one cell per cycle, with a dump_req gap every
    // 11 cells (back-pressure: data must be held and ordering preserved).
    task automatic dump_only(input int core, input int row0);
        int gap;
        for (int i = 0; i < ROWS*COLS; i++) begin
            @(negedge clk);
            if (i > 0) check_cell(core, row0, i-1);
            issue_dump(core, i);
            // Insert an idle cycle: the core must keep last read stable.
            if (i % 11 == 5) begin
                @(negedge clk);
                if (core == 0) begin dump_req0 = 0; end else begin dump_req1 = 0; end
                check_cell(core, row0, i);
                // re-arm so the loop's next iteration request is the next cell
                issue_dump(core, i);
            end
        end
        @(negedge clk);
        check_cell(core, row0, ROWS*COLS-1);
        dump_req0 = 0; dump_req1 = 0;
    endtask

    // True overlap: accumulate into `sc` while exporting from `dc`.
    task automatic overlap_both(input int sc, input int srow0,
                                input int dc, input int drow0);
        for (int i = 0; i < ROWS*COLS; i++) begin
            @(negedge clk);
            in_valid0 = 0; in_valid1 = 0;
            stream_step(sc, i, srow0);
            if (i > 0) check_cell(dc, drow0, i-1);
            issue_dump(dc, i);
        end
        @(negedge clk);
        in_valid0 = 0; in_valid1 = 0;
        check_cell(dc, drow0, ROWS*COLS-1);
        dump_req0 = 0; dump_req1 = 0;
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

        // Reset check: both pages start invalid.
        repeat (4) @(negedge clk);
        if (dump_valid0 !== 1'b0 || dump_valid1 !== 1'b0) begin
            $error("dump_valid not cleared by reset"); errors++;
        end
        @(negedge clk);
        rst_n = 1'b1;
        row_mask0 = {ROWS{1'b1}}; row_mask1 = {ROWS{1'b1}};
        @(negedge clk);

        // Tile 0 -> page0, exported later.
        stream_only(0, 0, 1'b0);
        // Tile 1 -> page1 while page0 is exported.
        overlap_both(1, ROWS, 0, 0);
        // Then export page1.
        dump_only(1, ROWS);

        // Mid-run reset must invalidate both pages (no stale accumulator).
        @(negedge clk);
        rst_n = 1'b0; dump_req0 = 0; dump_req1 = 0;
        repeat (2) @(negedge clk);
        if (dump_valid0 !== 1'b0 || dump_valid1 !== 1'b0) begin
            $error("dump_valid survived mid-test reset"); errors++;
        end
        rst_n = 1'b1;
        row_mask0 = {ROWS{1'b1}}; row_mask1 = {ROWS{1'b1}};
        stream_only(0, 0, 1'b0);
        dump_only(0, 0);

        if (errors != 0) $fatal(1, "dualacc unit FAILED with %0d errors", errors);
        $display("PASS dual-cacc unit: overlap dump+accumulate, reset, back-pressure ordering");
        $finish;
    end
endmodule
