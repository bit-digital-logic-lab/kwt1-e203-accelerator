// Compute-only regression: independent integer scoreboard, no storage/bus model.
module tb_kwt_mac_pipe3;
    localparam int ROWS = 16, COLS = 16;
    logic clk = 0, rst_n = 0, clear = 0, in_valid = 0;
    logic [ROWS-1:0] row_mask = '0;
    logic [ROWS*16-1:0] a_rows = '0;
    logic [15:0] b_word = '0;
    logic [3:0] in_col = '0;
    logic dump_req = 0, base_valid, pipe_valid;
    logic [3:0] dump_row = '0, dump_col = '0;
    logic [31:0] dump_bias = '0, base_data, pipe_data;
    logic signed [31:0] expected [0:ROWS-1][0:COLS-1];
    int accepted;

    always #5 clk = ~clk;
    kwt_row_broadcast_tile #(.ROWS(ROWS)) base (
        .clk(clk), .rst_n(rst_n), .clear(clear), .in_valid(in_valid),
        .row_mask(row_mask), .a_rows(a_rows), .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(base_valid), .dump_data(base_data)
    );
    kwt_row_broadcast_tile_pipe3 #(.ROWS(ROWS)) pipe3 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .in_valid(in_valid),
        .row_mask(row_mask), .a_rows(a_rows), .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(pipe_valid), .dump_data(pipe_data)
    );

    function automatic int col_for(input int pattern, input int t);
        case (pattern)
            0: return t % COLS;
            1: return 0;
            2: return t % 2;
            default: return ((t*37) ^ ((t/5)*13)) % COLS;
        endcase
    endfunction

    function automatic int signed sample(input int t, input int r);
        case ((t+r) % 8)
            0: return -32768;
            1: return 32767;
            2: return -1;
            3: return 0;
            default: return ((t*1103+r*7919) & 16'hffff) - 32768;
        endcase
    endfunction

    task automatic run_pattern(input int pattern, input int cycles);
        int av, bv, col;
        @(negedge clk);
        clear = 1;
        in_valid = 0;
        accepted = 0;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) expected[r][c] = 0;
        @(negedge clk);
        clear = 0;
        for (int t = 0; t < cycles; t++) begin
            col = col_for(pattern, t);
            bv = sample(t, 23);
            in_valid = (pattern != 3) || (t % 7 != 0);
            row_mask = (pattern != 3) ? '1 : 16'((t*117) ^ (t>>3) ^ 16'h9e37);
            in_col = 4'(col);
            b_word = 16'(bv);
            if (in_valid) accepted++;
            for (int r = 0; r < ROWS; r++) begin
                av = sample(t, r);
                a_rows[r*16 +: 16] = 16'(av);
                if (in_valid && row_mask[r])
                    expected[r][col] += 32'(av*bv);
            end
            @(negedge clk);
        end
        in_valid = 0;
        repeat (6) @(negedge clk);
        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                dump_req = 1;
                dump_row = 4'(r);
                dump_col = 4'(c);
                dump_bias = 32'((r*131 + c*43) - 777);
                @(negedge clk);
                if (!base_valid || !pipe_valid ||
                    base_data !== (expected[r][c] + dump_bias) ||
                    pipe_data !== (expected[r][c] + dump_bias))
                    $fatal(1, "pattern=%0d [%0d,%0d] base=%h pipe=%h expected=%h",
                           pattern, r, c, base_data, pipe_data,
                           expected[r][c] + dump_bias);
            end
        end
        dump_req = 0;
        $display("PASS pattern=%0d offered=%0d accepted=%0d", pattern, cycles, accepted);
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        run_pattern(0, 1024);
        run_pattern(1, 1024);
        run_pattern(2, 1024);
        run_pattern(3, 1024);
        $display("PASS pipe3 compute-only II=1 regression");
        $finish;
    end
endmodule
