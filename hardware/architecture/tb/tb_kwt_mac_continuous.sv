// Compute-only test: no A/W storage, DMA, output stage, or bus timing.
// It checks whether every asserted input cycle is accepted at II=1, including
// the hardest accumulation recurrences (same column on consecutive cycles).
module tb_kwt_mac_continuous;
    localparam int ROWS = 16, COLS = 16;
    logic clk = 0, rst_n = 0, clear = 0, in_valid = 0;
    logic [ROWS-1:0] row_mask = '1;
    logic [ROWS*16-1:0] a_rows = '0;
    logic [15:0] b_word = '0;
    logic [3:0] in_col = '0;
    logic dump_req = 0, dump_valid;
    logic [3:0] dump_row = '0, dump_col = '0;
    logic [31:0] dump_bias = '0, dump_data;
    logic signed [31:0] expected [0:ROWS-1][0:COLS-1];
    int accepted;

    always #5 clk = ~clk;
    kwt_row_broadcast_tile #(.ROWS(ROWS)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .in_valid(in_valid),
        .row_mask(row_mask), .a_rows(a_rows), .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(dump_valid), .dump_data(dump_data)
    );

    function automatic int select_col(input int pattern, input int t);
        case (pattern)
            0: return t % COLS;
            1: return 0;
            2: return t % 2;
            default: return ((t*37) ^ (t/5)*13) % COLS;
        endcase
    endfunction

    task automatic run_pattern(input int pattern, input int offered_cycles,
                               input bit insert_gaps);
        int col, av, bv, this_accepted;
        @(negedge clk);
        clear = 1;
        in_valid = 0;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) expected[r][c] = 0;
        accepted = 0;
        @(negedge clk);
        clear = 0;
        for (int t = 0; t < offered_cycles; t++) begin
            col = select_col(pattern, t);
            bv = ((t*11) % 97) - 48;
            in_valid = !insert_gaps || (t % 7 != 0);
            in_col = 4'(col);
            b_word = 16'(bv);
            if (in_valid) accepted++;
            for (int r = 0; r < ROWS; r++) begin
                av = ((r*7+t*3) % 101) - 50;
                a_rows[r*16 +: 16] = 16'(av);
                if (in_valid) expected[r][col] += 32'(av*bv);
            end
            @(negedge clk);
        end
        in_valid = 0;
        repeat (4) @(negedge clk);
        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                dump_req = 1;
                dump_row = 4'(r);
                dump_col = 4'(c);
                @(negedge clk);
                if (!dump_valid || dump_data !== expected[r][c])
                    $fatal(1, "pattern=%0d out[%0d,%0d]=%h expected=%h",
                           pattern, r, c, dump_data, expected[r][c]);
            end
        end
        dump_req = 0;
        this_accepted = accepted;
        $display("PASS pattern=%0d offered=%0d accepted=%0d useful_mac=%0d",
                 pattern, offered_cycles, this_accepted, this_accepted*ROWS);
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        run_pattern(0, 1024, 0); // round-robin columns, no input bubbles
        run_pattern(1, 1024, 0); // same column every cycle
        run_pattern(2, 1024, 0); // two columns with short recurrence distance
        run_pattern(3, 1024, 1); // irregular columns and valid gaps
        $display("PASS continuous 16-row MAC compute-only regression");
        $finish;
    end
endmodule
