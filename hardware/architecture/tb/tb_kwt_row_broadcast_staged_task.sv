// Task-level harness with an RTL CACC output stage. A/W/bias preload and the
// shared bus are still timed models; this isolates the output-overlap benefit.
module tb_kwt_row_broadcast_staged_task #(
    parameter int ROWS = 8,
    parameter int DATA_W = 16
);
    localparam int COLS = 16, ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS);
    localparam int MAX_M = 99, MAX_K = 64, MAX_N = 256;

    logic clk = 0, rst_n = 0;
    logic clear = 0, in_valid = 0, dump_req;
    logic [ROWS-1:0] row_mask = '0;
    logic [ROWS*DATA_W-1:0] a_rows = '0;
    logic [DATA_W-1:0] b_word = '0;
    logic [3:0] in_col = '0, dump_col;
    logic [ROW_W-1:0] dump_row;
    logic [31:0] dump_bias;
    logic dump_valid;
    logic [31:0] dump_data;
    logic stage_start_valid = 0, stage_start_ready;
    logic [6:0] stage_base_row, stage_m_len;
    logic [7:0] stage_base_col, stage_bias_col;
    logic [8:0] stage_n_len;
    logic [31:0] stage_bias_data;
    logic stage_capture_done, stage_out_valid, stage_out_ready, stage_done;
    logic [6:0] stage_out_row;
    logic [7:0] stage_out_col;
    logic [31:0] stage_out_data;

    logic [15:0] a_source [0:MAX_M*MAX_K-1];
    logic [15:0] w_source [0:MAX_K*MAX_N-1];
    logic [31:0] bias_source [0:MAX_N-1];
    logic [31:0] expected [0:MAX_M*MAX_N-1];
    logic [15:0] a_store [0:MAX_M*MAX_K-1];
    logic [15:0] w_page [0:1][0:MAX_K*COLS-1];
    logic [31:0] bias_store [0:MAX_N-1];
    logic [15:0] a_hold [0:ROWS-1];
    logic [15:0] a_next [0:ROWS-1];

    string a_path, w_path, bias_path, expected_path, vcd_path, output_path;
    integer output_fd = 0;
    int m_len, k_len, n_len, insert_gaps;
    int cycle_count = 0;
    int a_load_words = 0, w_load_words = 0, bias_load_words = 0;
    int a_local_reads = 0, w_local_reads = 0, out_write_words = 0;
    int checked_outputs = 0, checked_padding = 0;
    int initial_load_cycles = 0, compute_cycles = 0, drain_cycles = 0;
    int overlapped_output_words = 0;

    always #5 clk = ~clk;
    always @(posedge clk) if (rst_n) cycle_count <= cycle_count + 1;
    initial begin
        #10000000;
        $fatal(1, "staged task watchdog expired");
    end

    kwt_row_broadcast_tile #(.ROWS(ROWS), .DATA_W(DATA_W)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .in_valid(in_valid), .row_mask(row_mask),
        .a_rows(a_rows), .b_word(b_word), .in_col(in_col),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(dump_valid), .dump_data(dump_data)
    );
    always @(posedge clk) stage_bias_data <= bias_store[stage_bias_col];
    // One result consumes two 16-bit transfer slots. The slot pattern is an
    // abstract bus model; readiness may be delayed without losing data.
    assign stage_out_ready = cycle_count[0];
    kwt_tile_output_stage #(.ROWS(ROWS), .COLS(COLS)) stage (
        .clk(clk), .rst_n(rst_n), .start_valid(stage_start_valid),
        .start_ready(stage_start_ready), .base_row(stage_base_row),
        .base_col(stage_base_col), .m_len(stage_m_len), .n_len(stage_n_len),
        .bias_col(stage_bias_col), .bias_data(stage_bias_data),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_bias(dump_bias), .dump_valid(dump_valid), .dump_data(dump_data),
        .capture_done(stage_capture_done), .out_valid(stage_out_valid),
        .out_ready(stage_out_ready), .out_row(stage_out_row),
        .out_col(stage_out_col), .out_data(stage_out_data), .done(stage_done)
    );

    always @(posedge clk) if (rst_n && stage_out_valid && stage_out_ready) begin
        if (stage_out_data !== expected[int'(stage_out_row)*n_len+int'(stage_out_col)])
            $fatal(1, "staged output[%0d,%0d] got=%h expected=%h",
                   stage_out_row, stage_out_col, stage_out_data,
                   expected[int'(stage_out_row)*n_len+int'(stage_out_col)]);
        if (output_fd != 0)
            $fdisplay(output_fd, "%0d %0d %08h", stage_out_row,
                      stage_out_col, stage_out_data);
        checked_outputs <= checked_outputs + 1;
        out_write_words <= out_write_words + 2;
        if (in_valid) overlapped_output_words <= overlapped_output_words + 1;
    end

    task automatic load_weight_word(input int page, input int group, input int index);
        int k, c, global_col;
        if (stage_out_valid)
            $fatal(1, "shared word bus conflict: W preload during output writeback");
        k = index / COLS;
        c = index % COLS;
        global_col = group*COLS+c;
        if (global_col < n_len) begin
            w_page[page][index] = w_source[k*n_len+global_col];
            w_load_words++;
        end else w_page[page][index] = '0;
    endtask

    initial begin : run_task
        int row_tiles, col_tiles, active_page;
        int phase_start;
        if (!$value$plusargs("A=%s", a_path) ||
            !$value$plusargs("W=%s", w_path) ||
            !$value$plusargs("BIAS=%s", bias_path) ||
            !$value$plusargs("EXPECTED=%s", expected_path) ||
            !$value$plusargs("M=%d", m_len) ||
            !$value$plusargs("K=%d", k_len) ||
            !$value$plusargs("N=%d", n_len))
            $fatal(1, "missing task input");
        if (m_len < 1 || m_len > MAX_M || k_len < 1 || k_len > MAX_K ||
            n_len < 1 || n_len > MAX_N)
            $fatal(1, "unsupported task shape");
        if (ROWS != 8 && ROWS != 16)
            $fatal(1, "task harness supports ROWS=8 or ROWS=16");
        if (DATA_W != 8 && DATA_W != 16)
            $fatal(1, "task harness supports DATA_W=8 or DATA_W=16");
        insert_gaps = 0;
        if ($value$plusargs("GAPS=%d", insert_gaps)) begin end
        if ($value$plusargs("VCD=%s", vcd_path)) begin
            $dumpfile(vcd_path);
            $dumpvars(0, tb_kwt_row_broadcast_task);
        end
        if ($value$plusargs("OUTPUT=%s", output_path)) begin
            output_fd = $fopen(output_path, "w");
            if (output_fd == 0) $fatal(1, "cannot open output file: %s", output_path);
        end
        $readmemh(a_path, a_source);
        $readmemh(w_path, w_source);
        $readmemh(bias_path, bias_source);
        $readmemh(expected_path, expected);
        row_tiles = (m_len+ROWS-1)/ROWS;
        col_tiles = (n_len+COLS-1)/COLS;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        phase_start = cycle_count;

        // Task begins with an empty local A store and bias store.
        for (int i = 0; i < m_len*k_len; i++) begin
            @(negedge clk);
            a_store[i] = a_source[i];
            a_load_words++;
        end
        for (int n = 0; n < n_len; n++) begin
            repeat (2) begin
                @(negedge clk);
                bias_load_words++;
            end
            bias_store[n] = bias_source[n];
        end
        for (int i = 0; i < k_len*COLS; i++) begin
            @(negedge clk);
            load_weight_word(0, 0, i);
        end
        initial_load_cycles = cycle_count - phase_start;

        active_page = 0;
        for (int ct = 0; ct < col_tiles; ct++) begin
            int next_w_index;
            next_w_index = 0;
            for (int rt = 0; rt < row_tiles; rt++) begin
                @(negedge clk);
                clear = 1'b1;
                in_valid = 1'b0;
                @(negedge clk);
                clear = 1'b0;
                row_mask = '0;
                for (int r = 0; r < ROWS; r++)
                    row_mask[r] = (rt*ROWS+r < m_len);

                // First A vector takes ROWS one-word reads. Later vectors
                // are fetched during the current 16-column B stream.
                phase_start = cycle_count;
                for (int r = 0; r < ROWS; r++) begin
                    @(negedge clk);
                    if (row_mask[r]) begin
                        a_hold[r] = a_store[(rt*ROWS+r)*k_len];
                        a_local_reads++;
                    end else a_hold[r] = '0;
                end

                for (int k = 0; k < k_len; k++) begin
                    for (int c = 0; c < COLS; c++) begin
                        if (insert_gaps != 0 && (k*COLS+c) % 17 == 0) begin
                            @(negedge clk);
                            in_valid = 1'b0;
                        end
                        @(negedge clk);
                        in_valid = 1'b1;
                        in_col = 4'(c);
                        b_word = w_page[active_page][k*COLS+c][DATA_W-1:0];
                        w_local_reads++;
                        for (int r = 0; r < ROWS; r++)
                            a_rows[r*DATA_W +: DATA_W] = a_hold[r][DATA_W-1:0];
                        if (c < ROWS && k+1 < k_len) begin
                            if (row_mask[c]) begin
                                a_next[c] = a_store[(rt*ROWS+c)*k_len+k+1];
                                a_local_reads++;
                            end else a_next[c] = '0;
                        end
                        // Physical W pages are separate: compute reads this
                        // page while the next group fills the other page.
                        if (ct+1 < col_tiles && next_w_index < k_len*COLS &&
                            !stage_out_valid) begin
                            load_weight_word(1-active_page, ct+1, next_w_index);
                            next_w_index++;
                        end
                    end
                    if (k+1 < k_len)
                        for (int r = 0; r < ROWS; r++) a_hold[r] = a_next[r];
                end
                @(negedge clk);
                in_valid = 1'b0;
                repeat (4) @(negedge clk);
                compute_cycles += cycle_count - phase_start;
                phase_start = cycle_count;
                while (!stage_start_ready) @(negedge clk);
                stage_base_row = 7'(rt*ROWS);
                stage_base_col = 8'(ct*COLS);
                stage_m_len = 7'(m_len);
                stage_n_len = 9'(n_len);
                stage_start_valid = 1'b1;
                @(negedge clk);
                stage_start_valid = 1'b0;
                wait (stage_capture_done);
                @(negedge clk);
                drain_cycles += cycle_count - phase_start;
            end
            // Finish the next page only in word slots not used by output
            // writeback. This matters for tasks with very few row tiles.
            while (ct+1 < col_tiles && next_w_index < k_len*COLS) begin
                @(negedge clk);
                if (!stage_out_valid) begin
                    load_weight_word(1-active_page, ct+1, next_w_index);
                    next_w_index++;
                end
            end
            if (ct+1 < col_tiles && next_w_index != k_len*COLS)
                $fatal(1, "next W page incomplete: %0d/%0d", next_w_index, k_len*COLS);
            active_page = 1-active_page;
        end

        while (!stage_start_ready || checked_outputs != m_len*n_len)
            @(negedge clk);

        if (a_load_words != m_len*k_len ||
            w_load_words != k_len*n_len ||
            bias_load_words != 2*n_len ||
            out_write_words != 2*m_len*n_len ||
            a_local_reads != m_len*k_len*col_tiles ||
            w_local_reads != row_tiles*col_tiles*k_len*COLS ||
            checked_outputs != m_len*n_len)
            $fatal(1, "architectural access count mismatch");
        $display("ARCH_METRICS cycles=%0d initial_load_cycles=%0d compute_cycles=%0d drain_cycles=%0d a_load_words=%0d w_load_words=%0d bias_load_words=%0d out_write_words=%0d a_local_reads=%0d w_local_reads=%0d checked_outputs=%0d checked_padding=%0d overlapped_output_words=%0d",
                 cycle_count, initial_load_cycles, compute_cycles, drain_cycles,
                 a_load_words, w_load_words, bias_load_words,
                 out_write_words, a_local_reads, w_local_reads,
                 checked_outputs, checked_padding, overlapped_output_words);
        $display("PASS row-broadcast staged task M=%0d K=%0d N=%0d gaps=%0d",
                 m_len, k_len, n_len, insert_gaps);
        if (output_fd != 0) $fclose(output_fd);
        $finish;
    end
endmodule
