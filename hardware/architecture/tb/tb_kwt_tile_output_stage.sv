module tb_kwt_tile_output_stage;
    localparam int ROWS = 16, COLS = 16;
    logic clk = 0, rst_n = 0;
    logic clear = 0, in_valid = 0;
    logic [ROWS-1:0] row_mask = '0;
    logic [ROWS*16-1:0] a_rows = '0;
    logic [15:0] b_word = '0;
    logic [3:0] in_col = '0;
    logic start_valid = 0, start_ready;
    logic [6:0] base_row = 0, m_len = 0;
    logic [7:0] base_col = 0, bias_col;
    logic [8:0] n_len = 0;
    logic [31:0] bias_data;
    logic dump_req, dump_valid, capture_done;
    logic [3:0] dump_row, dump_col;
    logic [31:0] dump_bias, dump_data;
    logic out_valid, out_ready = 0, done;
    logic [6:0] out_row;
    logic [7:0] out_col;
    logic [31:0] out_data;
    int cycles = 0, seen = 0, want_count = 0;
    int expected_base_row, expected_base_col, expected_cols;
    logic stalled_q = 0;
    logic [6:0] stalled_row;
    logic [7:0] stalled_col;
    logic [31:0] stalled_data;

    always #5 clk = ~clk;
    always @(posedge clk) bias_data <= 32'(100 + int'(bias_col));
    always @(negedge clk) if (rst_n) out_ready = (cycles % 3 != 0);

    kwt_row_broadcast_tile #(.ROWS(ROWS)) core (
        .clk(clk), .rst_n(rst_n), .clear(clear), .in_valid(in_valid),
        .row_mask(row_mask), .a_rows(a_rows), .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(dump_valid), .dump_data(dump_data)
    );
    kwt_tile_output_stage #(.ROWS(ROWS), .COLS(COLS)) stage (
        .clk(clk), .rst_n(rst_n), .start_valid(start_valid),
        .start_ready(start_ready), .base_row(base_row), .base_col(base_col),
        .m_len(m_len), .n_len(n_len), .bias_col(bias_col),
        .bias_data(bias_data), .dump_req(dump_req), .dump_row(dump_row),
        .dump_col(dump_col), .dump_bias(dump_bias), .dump_valid(dump_valid),
        .dump_data(dump_data), .capture_done(capture_done),
        .out_valid(out_valid), .out_ready(out_ready), .out_row(out_row),
        .out_col(out_col), .out_data(out_data), .done(done)
    );

    function automatic logic [31:0] expected(input int row, input int col);
        int accum;
        accum = 100 + col;
        for (int k = 0; k < 3; k++)
            accum += (row - 8 + k) * (col - 8 - k);
        return 32'(accum);
    endfunction

    always @(posedge clk) if (rst_n) begin
        cycles <= cycles + 1;
        if (stalled_q && (!out_valid || out_row !== stalled_row ||
                          out_col !== stalled_col || out_data !== stalled_data))
            $fatal(1, "output changed under backpressure");
        stalled_q <= out_valid && !out_ready;
        if (out_valid && !out_ready) begin
            stalled_row <= out_row;
            stalled_col <= out_col;
            stalled_data <= out_data;
        end
        if (out_valid && out_ready) begin
            if (int'(out_row) != expected_base_row + seen / expected_cols ||
                int'(out_col) != expected_base_col + seen % expected_cols ||
                out_data !== expected(int'(out_row), int'(out_col)))
                $fatal(1, "out[%0d,%0d]=%h expected=%h seen=%0d",
                       out_row, out_col, out_data,
                       expected(int'(out_row), int'(out_col)), seen);
            seen <= seen + 1;
        end
    end

    task automatic run_tile(input int br, input int bc, input int m, input int n);
        @(negedge clk);
        if (!start_ready) $fatal(1, "stage was not empty before tile");
        clear = 1;
        in_valid = 0;
        base_row = 7'(br);
        base_col = 8'(bc);
        m_len = 7'(m);
        n_len = 9'(n);
        expected_base_row = br;
        expected_base_col = bc;
        expected_cols = n-bc;
        want_count = (m-br)*(n-bc);
        seen = 0;
        for (int r = 0; r < ROWS; r++) row_mask[r] = (br+r < m);
        @(negedge clk);
        clear = 0;
        for (int k = 0; k < 3; k++) begin
            for (int c = 0; c < COLS; c++) begin
                in_valid = 1;
                in_col = 4'(c);
                b_word = 16'(bc+c-8-k);
                for (int r = 0; r < ROWS; r++)
                    a_rows[r*16 +: 16] = 16'(br+r-8+k);
                @(negedge clk);
            end
        end
        in_valid = 0;
        repeat (4) @(negedge clk);
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;
        wait (capture_done);
        wait (done);
        @(negedge clk);
        if (seen != want_count)
            $fatal(1, "tile (%0d,%0d) outputs=%0d expected=%0d",
                   br, bc, seen, want_count);
        $display("PASS output stage tile=(%0d,%0d) outputs=%0d cycles=%0d",
                 br, bc, seen, cycles);
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        run_tile(0, 0, 16, 16);
        run_tile(16, 16, 19, 21);
        $display("PASS output stage backpressure and edge masking");
        $finish;
    end
endmodule
