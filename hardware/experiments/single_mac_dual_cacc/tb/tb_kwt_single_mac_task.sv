// Task-level harness for the ONE-MAC-ARRAY / TWO-CACC-PAGE dataflow.
//
// It instantiates exactly ONE kwt_single_mac_dual_cacc (16 shared multipliers)
// and the UNCHANGED kwt_tile_output_stage. The schedule is the same tile
// schedule measured by hardware/experiments/streaming_mac: the CACC export of
// tile T runs under the compute window of tile T+1, alternating two accumulator
// pages *inside* the single MAC array instead of instantiating the old core
// twice.
//
// The old RTL is only used for comparison by run_single_mac_dual_cacc.py
// (old one-core baseline + old two-core dual), never by this harness.
//
// A/W/bias preload, the shared 16-bit word bus and the output bus are still
// timed models (one word per cycle for load, one 16-bit slot per cycle with a
// configurable output-ready duty for store). It is NOT an E203/DMA/SRAM-macro
// or 150 MHz result.
module tb_kwt_single_mac_task #(
    parameter int ROWS = 16,
    parameter int DATA_W = 16
);
    localparam int COLS = 16, ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS);
    localparam int PAGE_W = 1;
    localparam int MAX_M = 99, MAX_K = 64, MAX_N = 256;

    logic clk = 0, rst_n = 0;
    logic clear_bit = 0, stream_valid = 0;
    logic [ROWS-1:0] rm = '0;
    logic [ROWS*DATA_W-1:0] ar = '0;
    logic [DATA_W-1:0] bw = '0;
    logic [3:0] ic = '0;
    int cur = 0;

    logic [PAGE_W-1:0] in_page;
    logic [1:0] clear_pages;
    logic in_ready;
    logic dump_req;
    logic dump_ready;
    logic dump_valid;
    logic [31:0] dump_data;

    logic stage_dump_req;
    logic [ROW_W-1:0] stage_dump_row;
    logic [3:0] stage_dump_col;
    logic [31:0] stage_dump_bias;
    logic cap_sel = 0;

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
    int m_len, k_len, n_len, insert_gaps, bp_period;
    int cycle_count = 0;
    int a_load_words = 0, w_load_words = 0, bias_load_words = 0;
    int a_local_reads = 0, w_local_reads = 0, out_write_words = 0;
    int checked_outputs = 0, checked_padding = 0;
    int initial_load_cycles = 0, compute_cycles = 0, drain_cycles = 0;
    int stall_cycles = 0, overlapped_output_words = 0, overlapped_captures = 0;
    int feed_stall_cycles = 0, dump_not_ready_cycles = 0, dump_beats = 0;

    always #5 clk = ~clk;
    always @(posedge clk) if (rst_n) cycle_count <= cycle_count + 1;
    initial begin
        #200000000;
        $fatal(1, "single-mac task watchdog expired");
    end

    // Page routing: the compute stream selects one CACC page per beat.
    assign in_page = PAGE_W'(cur);
    assign clear_pages = clear_bit ? (cur ? 2'b10 : 2'b01) : 2'b00;

    // The ONE shared multiplier array + TWO accumulator pages.
    kwt_single_mac_dual_cacc #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W)) core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(stream_valid), .in_ready(in_ready), .in_page(in_page),
        .row_mask(rm), .a_rows(ar), .b_word(bw), .in_col(ic),
        .clear(clear_pages),
        .dump_req(stage_dump_req), .dump_ready(dump_ready),
        .dump_page(PAGE_W'(cap_sel)), .dump_row(stage_dump_row),
        .dump_col(stage_dump_col), .dump_bias(stage_dump_bias),
        .dump_valid(dump_valid), .dump_valid_page(), .dump_valid_row(),
        .dump_valid_col(), .dump_addr(), .dump_data(dump_data)
    );

    always @(posedge clk) stage_bias_data <= bias_store[stage_bias_col];
    // Abstract store bus: one result (two 16-bit slots) is accepted at the
    // configured duty cycle. Larger periods = deeper backpressure.
    assign stage_out_ready = (cycle_count % bp_period) == 0;

    kwt_tile_output_stage #(.ROWS(ROWS), .COLS(COLS)) stage (
        .clk(clk), .rst_n(rst_n), .start_valid(stage_start_valid),
        .start_ready(stage_start_ready), .base_row(stage_base_row),
        .base_col(stage_base_col), .m_len(stage_m_len), .n_len(stage_n_len),
        .bias_col(stage_bias_col), .bias_data(stage_bias_data),
        .dump_req(stage_dump_req), .dump_row(stage_dump_row), .dump_col(stage_dump_col),
        .dump_bias(stage_dump_bias),
        .dump_valid(dump_valid), .dump_data(dump_data),
        .capture_done(stage_capture_done), .out_valid(stage_out_valid),
        .out_ready(stage_out_ready), .out_row(stage_out_row),
        .out_col(stage_out_col), .out_data(stage_out_data), .done(stage_done)
    );

    // Dump request monitoring: the output stage is an accepted RTL block and
    // assumes its request is always ready. Record any violation honestly.
    always @(posedge clk) if (rst_n && stage_dump_req) begin
        dump_beats <= dump_beats + 1;
        if (!dump_ready) dump_not_ready_cycles <= dump_not_ready_cycles + 1;
    end

    always @(posedge clk) if (rst_n && stage_out_valid && stage_out_ready) begin
        if (stage_out_data !== expected[int'(stage_out_row)*n_len+int'(stage_out_col)])
            $fatal(1, "single-mac output[%0d,%0d] got=%h expected=%h",
                   stage_out_row, stage_out_col, stage_out_data,
                   expected[int'(stage_out_row)*n_len+int'(stage_out_col)]);
        if (output_fd != 0)
            $fdisplay(output_fd, "%0d %0d %08h", stage_out_row,
                      stage_out_col, stage_out_data);
        checked_outputs <= checked_outputs + 1;
        out_write_words <= out_write_words + 2;
        if (stream_valid) overlapped_output_words <= overlapped_output_words + 1;
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

    task automatic start_capture(input int page, input int base_row, input int base_col);
        int t0;
        t0 = cycle_count;
        while (!stage_start_ready) @(negedge clk);
        if (cycle_count != t0) stall_cycles += cycle_count - t0;
        cap_sel = page[0];
        stage_base_row = 7'(base_row);
        stage_base_col = 8'(base_col);
        stage_m_len = 7'(m_len);
        stage_n_len = 9'(n_len);
        stage_start_valid = 1'b1;
        @(negedge clk);
        stage_start_valid = 1'b0;
    endtask

    initial begin : run_task
        int row_tiles, col_tiles, active_page, tile_index, pending;
        int pending_row, pending_col;
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
        insert_gaps = 0;
        if ($value$plusargs("GAPS=%d", insert_gaps)) begin end
        bp_period = 2;
        if ($value$plusargs("BACKPRESSURE=%d", bp_period)) begin end
        if (bp_period < 1) $fatal(1, "BACKPRESSURE must be >= 1");
        if ($value$plusargs("VCD=%s", vcd_path)) begin
            $dumpfile(vcd_path);
            $dumpvars(0, tb_kwt_single_mac_task);
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
        pending = -1;
        tile_index = 0;
        for (int ct = 0; ct < col_tiles; ct++) begin
            int next_w_index;
            next_w_index = 0;
            for (int rt = 0; rt < row_tiles; rt++) begin
                cur = tile_index % 2;
                if (pending >= 0) begin
                    start_capture(pending, pending_row, pending_col);
                    overlapped_captures++;
                    pending = -1;
                end

                @(negedge clk);
                clear_bit = 1'b1;
                stream_valid = 1'b0;
                @(negedge clk);
                clear_bit = 1'b0;
                rm = '0;
                for (int r = 0; r < ROWS; r++)
                    rm[r] = (rt*ROWS+r < m_len);

                phase_start = cycle_count;
                for (int r = 0; r < ROWS; r++) begin
                    @(negedge clk);
                    if (rm[r]) begin
                        a_hold[r] = a_store[(rt*ROWS+r)*k_len];
                        a_local_reads++;
                    end else a_hold[r] = '0;
                end

                for (int k = 0; k < k_len; k++) begin
                    for (int c = 0; c < COLS; c++) begin
                        if (insert_gaps != 0 && (k*COLS+c) % 17 == 0) begin
                            @(negedge clk);
                            stream_valid = 1'b0;
                        end
                        @(negedge clk);
                        stream_valid = 1'b1;
                        ic = 4'(c);
                        bw = w_page[active_page][k*COLS+c][DATA_W-1:0];
                        w_local_reads++;
                        for (int r = 0; r < ROWS; r++)
                            ar[r*DATA_W +: DATA_W] = a_hold[r][DATA_W-1:0];
                        if (c < ROWS && k+1 < k_len) begin
                            if (rm[c]) begin
                                a_next[c] = a_store[(rt*ROWS+c)*k_len+k+1];
                                a_local_reads++;
                            end else a_next[c] = '0;
                        end
                        if (ct+1 < col_tiles && next_w_index < k_len*COLS &&
                            !stage_out_valid) begin
                            load_weight_word(1-active_page, ct+1, next_w_index);
                            next_w_index++;
                        end
                        // Feed valid/ready handshake: hold until the single MAC
                        // array can accept the beat (export conflict / clear).
                        #1;
                        while (!in_ready) begin
                            feed_stall_cycles++;
                            @(negedge clk);
                            #1;
                        end
                    end
                    if (k+1 < k_len)
                        for (int r = 0; r < ROWS; r++) a_hold[r] = a_next[r];
                end
                @(negedge clk);
                stream_valid = 1'b0;
                repeat (4) @(negedge clk);
                compute_cycles += cycle_count - phase_start;
                pending = cur;
                pending_row = rt*ROWS;
                pending_col = ct*COLS;
                tile_index++;
            end
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

        phase_start = cycle_count;
        if (pending >= 0) begin
            start_capture(pending, pending_row, pending_col);
            pending = -1;
        end
        while (!stage_start_ready || checked_outputs != m_len*n_len)
            @(negedge clk);
        drain_cycles = cycle_count - phase_start;

        if (a_load_words != m_len*k_len ||
            w_load_words != k_len*n_len ||
            bias_load_words != 2*n_len ||
            out_write_words != 2*m_len*n_len ||
            a_local_reads != m_len*k_len*col_tiles ||
            w_local_reads != row_tiles*col_tiles*k_len*COLS ||
            checked_outputs != m_len*n_len)
            $fatal(1, "architectural access count mismatch");
        if (dump_not_ready_cycles != 0)
            $fatal(1, "output-stage dump request was not ready %0d cycles", dump_not_ready_cycles);
        $display("ARCH_METRICS cycles=%0d initial_load_cycles=%0d compute_cycles=%0d drain_cycles=%0d stall_cycles=%0d feed_stall_cycles=%0d dump_not_ready_cycles=%0d a_load_words=%0d w_load_words=%0d bias_load_words=%0d out_write_words=%0d a_local_reads=%0d w_local_reads=%0d checked_outputs=%0d checked_padding=%0d overlapped_output_words=%0d overlapped_captures=%0d",
                 cycle_count, initial_load_cycles, compute_cycles, drain_cycles,
                 stall_cycles, feed_stall_cycles, dump_not_ready_cycles,
                 a_load_words, w_load_words, bias_load_words,
                 out_write_words, a_local_reads, w_local_reads,
                 checked_outputs, checked_padding, overlapped_output_words,
                 overlapped_captures);
        $display("PASS single-mac-array dual-cacc task M=%0d K=%0d N=%0d gaps=%0d bp=%0d",
                 m_len, k_len, n_len, insert_gaps, bp_period);
        if (output_fd != 0) $fclose(output_fd);
        $finish;
    end
endmodule
