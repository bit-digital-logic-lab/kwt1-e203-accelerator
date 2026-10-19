// One-descriptor FFN1 task harness with concurrent A / W / bias fill and
// bias-at-CACC-export. This is a *new* task TB; the arithmetic core and output
// stage are the accepted, unchanged RTL.
//
// Schedule (one descriptor, no serial LOAD_BIAS state):
//   * A resident bank is filled sequentially; a row tile may compute as soon
//     as its own A rows are resident.  +A_PRELOAD_TILES selects how many row
//     tiles are preloaded before the first tile (default: all).
//   * W is double-buffered by column group.  The next group is prefetched
//     while the current group computes.
//   * Bias is filled into a dedicated 256x32 bank in column groups of 16
//     INT32 values = 32 shared 16-bit slots.  Bias is NOT a pre-MAC barrier:
//     compute starts without it.  An export waits only if its group is not
//     valid yet, and the core adds dump_bias at CACC dump.
//   * Two CACC pages alternate: MAC on page p while page 1-p is exported.
//
// Bus contract: one 16-bit slot per cycle. OUT > BIAS > A > W is a fixed
// priority; each requester holds its request/data until gnt. The output stage
// consumes two slots per INT32 result. No signal in this TB asserts that two
// requesters can move data on the shared bus in the same cycle.
//
// This is a functional/task-level model, not an E203 run and not 150 MHz STA.
module tb_kwt_bias_overlap_task #(
    parameter int ROWS = 16,
    parameter int DATA_W = 16
);
    localparam int COLS = 16;
    localparam int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS);
    localparam int MAX_M = 99, MAX_K = 64, MAX_N = 256;
    localparam int GROUPS = MAX_N / COLS;
    localparam int REQ_OUT = 0, REQ_BIAS = 1, REQ_A = 2, REQ_W = 3;

    logic clk = 0, rst_n = 0;
    logic clear_bit = 0, stream_valid = 0;
    logic [ROWS-1:0] rm = '0;
    logic [ROWS*DATA_W-1:0] ar = '0;
    logic [DATA_W-1:0] bw = '0;
    logic [3:0] ic = '0;
    int cur = 0;

    logic clear0, in_valid0, clear1, in_valid1;
    logic [ROWS-1:0] row_mask0, row_mask1;
    logic [ROWS*DATA_W-1:0] a_rows0, a_rows1;
    logic [DATA_W-1:0] b_word0, b_word1;
    logic [3:0] in_col0, in_col1;
    logic dump_req0, dump_req1;
    logic dump_valid0, dump_valid1;
    logic [31:0] dump_data0, dump_data1;

    logic stage_dump_req;
    logic [ROW_W-1:0] stage_dump_row;
    logic [3:0] stage_dump_col;
    logic [31:0] stage_dump_bias;
    logic dump_valid;
    logic [31:0] dump_data;
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

    // Dedicated bias bank and shared-bus arbiter (both new, synthesizable RTL).
    logic bias_wr_en = 0;
    logic [3:0] bias_wr_group = 0;
    logic [3:0] bias_wr_idx = 0;
    logic bias_wr_half = 0;
    logic [15:0] bias_wr_data = 0;
    logic [7:0] bias_rd_col;
    logic [31:0] bias_rd_data;
    logic [GROUPS-1:0] bias_group_valid;

    logic [3:0] bus_req, bus_gnt;
    logic bus_enable = 1;
    logic bias_clear = 0;

    logic [15:0] a_source [0:MAX_M*MAX_K-1];
    logic [15:0] w_source [0:MAX_K*MAX_N-1];
    logic [31:0] bias_source [0:MAX_N-1];
    logic [31:0] expected [0:MAX_M*MAX_N-1];
    logic [15:0] a_store [0:MAX_M*MAX_K-1];
    logic [15:0] w_page [0:1][0:MAX_K*COLS-1];
    logic [15:0] a_hold [0:ROWS-1];
    logic [15:0] a_next [0:ROWS-1];

    string a_path, w_path, bias_path, expected_path, vcd_path, output_path;
    integer output_fd = 0;
    int m_len, k_len, n_len, insert_gaps;
    int row_tiles, col_tiles, total_tiles;
    int cycle_count = 0;
    int task_start_cycle = 0;
    int a_load_words = 0, w_load_words = 0, w_fetch_words = 0, bias_load_words = 0;
    int a_local_reads = 0, w_local_reads = 0, out_write_words = 0;
    int checked_outputs = 0, checked_padding = 0;
    int initial_load_cycles = 0, compute_wall_cycles = 0, tail_cycles = 0;
    int cacc_stall_cycles = 0, load_stall_cycles = 0, bias_stall_cycles = 0;
    int stage_stall_cycles = 0, out_bp_stall_cycles = 0;
    int overlapped_captures = 0, overlapped_output_words = 0;
    int bus_grant_out = 0, bus_grant_bias = 0, bus_grant_a = 0, bus_grant_w = 0;
    int bus_idle_cycles = 0, bus_cycles = 0;

    // A/W/bias loader state.
    int a_loaded = 0;
    int a_preload_tiles = -1;
    int a_preload_words = 0;
    int w_group = 0, w_idx = 0;
    logic w_active = 0;
    logic [1:0] w_page_valid = '0;
    logic [1:0] w_page_busy = '0;
    int w_page_group [0:1] = '{0, 0};
    logic [GROUPS-1:0] w_group_done = '0;
    int bias_group = 0, bias_idx = 0;
    logic bias_half = 0, bias_active = 0;
    int bias_delay = 0;
    int bp_period = 1;
    int bus_gap = 0;
    int compute_col = 0;
    int debug = 0;

    // Compute state.
    int comp_tile = 0, comp_phase = 0, comp_ct = 0, comp_rt = 0;
    int comp_page = 0, comp_row_idx = 0, mac_i = 0, tail_i = 0;
    logic compute_started = 0, compute_done = 0;
    int compute_start_cycle = 0, compute_done_cycle = 0;
    logic gap_pending = 0;

    // CACC page/export state.
    logic [1:0] cacc_busy = '0;
    logic [1:0] pending_valid = '0;
    int pending_tile [0:1] = '{0, 0};
    int pending_row [0:1] = '{0, 0};
    int pending_col [0:1] = '{0, 0};
    int pending_ct [0:1] = '{0, 0};
    int exp_tile = 0;
    logic exp_busy = 0;
    int exp_page = 0;

    // Output slot state: one INT32 result = two 16-bit slots.
    logic out_slot_phase = 0;
    logic out_accept = 0;
    logic out_slot_req;
    int out_slots = 0;
    int out_results = 0;

    always #5 clk = ~clk;
    always @(posedge clk) if (rst_n) cycle_count <= cycle_count + 1;
    initial begin
        #200000000;
        $fatal(1, "bias-overlap task watchdog expired");
    end

    // Route the compute stream to the selected CACC page.
    always_comb begin
        clear0 = 1'b0; in_valid0 = 1'b0; row_mask0 = '0;
        a_rows0 = '0; b_word0 = '0; in_col0 = '0;
        clear1 = 1'b0; in_valid1 = 1'b0; row_mask1 = '0;
        a_rows1 = '0; b_word1 = '0; in_col1 = '0;
        if (cur == 0) begin
            clear0 = clear_bit; in_valid0 = stream_valid; row_mask0 = rm;
            a_rows0 = ar; b_word0 = bw; in_col0 = ic;
        end else begin
            clear1 = clear_bit; in_valid1 = stream_valid; row_mask1 = rm;
            a_rows1 = ar; b_word1 = bw; in_col1 = ic;
        end
    end

    kwt_row_broadcast_tile #(.ROWS(ROWS), .DATA_W(DATA_W)) core0 (
        .clk(clk), .rst_n(rst_n), .clear(clear0),
        .in_valid(in_valid0), .row_mask(row_mask0),
        .a_rows(a_rows0), .b_word(b_word0), .in_col(in_col0),
        .dump_req(dump_req0), .dump_row(stage_dump_row), .dump_col(stage_dump_col),
        .dump_bias(stage_dump_bias), .dump_valid(dump_valid0), .dump_data(dump_data0)
    );
    kwt_row_broadcast_tile #(.ROWS(ROWS), .DATA_W(DATA_W)) core1 (
        .clk(clk), .rst_n(rst_n), .clear(clear1),
        .in_valid(in_valid1), .row_mask(row_mask1),
        .a_rows(a_rows1), .b_word(b_word1), .in_col(in_col1),
        .dump_req(dump_req1), .dump_row(stage_dump_row), .dump_col(stage_dump_col),
        .dump_bias(stage_dump_bias), .dump_valid(dump_valid1), .dump_data(dump_data1)
    );
    assign dump_req0 = stage_dump_req & ~cap_sel;
    assign dump_req1 = stage_dump_req &  cap_sel;
    assign dump_valid = cap_sel ? dump_valid1 : dump_valid0;
    assign dump_data  = cap_sel ? dump_data1  : dump_data0;

    kwt_bias_bank #(
        .DEPTH(MAX_N), .COLS(COLS), .GROUPS(GROUPS)
    ) bias_bank (
        .clk(clk), .rst_n(rst_n), .clear(bias_clear),
        .wr_en(bias_wr_en), .wr_group(bias_wr_group), .wr_idx(bias_wr_idx),
        .wr_half(bias_wr_half), .wr_data(bias_wr_data),
        .rd_col(bias_rd_col), .rd_data(bias_rd_data),
        .group_valid(bias_group_valid)
    );
    assign bias_rd_col = stage_bias_col;
    always @(posedge clk) if (rst_n) stage_bias_data <= bias_rd_data;

    kwt_shared_bus_arbiter #(.N(4)) arb (
        .clk(clk), .rst_n(rst_n), .enable(bus_enable),
        .req(bus_req), .gnt(bus_gnt)
    );

    // Output stage. Each accepted INT32 consumes two slots; out_accept is the
    // second-slot grant, so stage_out_ready is a single-cycle pulse.
    assign stage_out_ready = out_accept;
    assign out_slot_req = stage_out_valid &&
                          ((bp_period <= 1) || ((cycle_count % bp_period) == 0));

    // Request generation: one request per requester, held until grant.
    assign bus_req[REQ_OUT] = out_slot_req;
    assign bus_req[REQ_BIAS] = bias_active && (bias_group < col_tiles) &&
                               (cycle_count >= bias_delay);
    assign bus_req[REQ_A] = (a_loaded < a_preload_words) ||
                            (compute_started && (a_loaded < m_len * k_len));
    assign bus_req[REQ_W] = w_active;

    kwt_tile_output_stage #(.ROWS(ROWS), .COLS(COLS)) stage (
        .clk(clk), .rst_n(rst_n), .start_valid(stage_start_valid),
        .start_ready(stage_start_ready), .base_row(stage_base_row),
        .base_col(stage_base_col), .m_len(stage_m_len), .n_len(stage_n_len),
        .bias_col(stage_bias_col), .bias_data(stage_bias_data),
        .dump_req(stage_dump_req), .dump_row(stage_dump_row), .dump_col(stage_dump_col),
        .dump_bias(stage_dump_bias), .dump_valid(dump_valid), .dump_data(dump_data),
        .capture_done(stage_capture_done), .out_valid(stage_out_valid),
        .out_ready(stage_out_ready), .out_row(stage_out_row),
        .out_col(stage_out_col), .out_data(stage_out_data), .done(stage_done)
    );

    // Output verification: independent INT32 expected file.
    always @(posedge clk) if (rst_n && stage_out_valid && stage_out_ready) begin
        if (stage_out_data !== expected[int'(stage_out_row)*n_len+int'(stage_out_col)])
            $fatal(1, "bias-overlap output[%0d,%0d] got=%h expected=%h",
                   stage_out_row, stage_out_col, stage_out_data,
                   expected[int'(stage_out_row)*n_len+int'(stage_out_col)]);
        if (output_fd != 0)
            $fdisplay(output_fd, "%0d %0d %08h", stage_out_row,
                      stage_out_col, stage_out_data);
        out_write_words <= out_write_words + 2;
        checked_outputs <= checked_outputs + 1;
        out_results <= out_results + 1;
        if (compute_started && !compute_done)
            overlapped_output_words <= overlapped_output_words + 1;
    end

    function automatic bit a_row_ready(input int rt);
        int need_rows;
        need_rows = (rt + 1) * ROWS;
        if (need_rows > m_len) need_rows = m_len;
        return a_loaded >= need_rows * k_len;
    endfunction

    function automatic bit w_page_ready(input int ct);
        int p;
        p = ct % 2;
        return w_page_valid[p] && (w_page_group[p] == ct);
    endfunction

    function automatic bit w_page_is_free(input int p);
        if (w_page_busy[p]) return 1'b0;
        if (!w_page_valid[p]) return 1'b1;
        return w_group_done[w_page_group[p]];
    endfunction

    initial begin : run_task
        int grant, kk, cc, r, g, p, next_tile, next_ct, next_rt;
        int a_preload_arg, busy_cycles;
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
        bp_period = 1;
        if ($value$plusargs("OUT_BP=%d", bp_period)) begin end
        if (bp_period < 1) $fatal(1, "OUT_BP must be >= 1");
        bus_gap = 0;
        if ($value$plusargs("BUS_GAP=%d", bus_gap)) begin end
        bias_delay = 0;
        if ($value$plusargs("BIAS_DELAY=%d", bias_delay)) begin end
        a_preload_arg = -1;
        if ($value$plusargs("A_PRELOAD_TILES=%d", a_preload_arg)) begin end
        debug = 0;
        if ($value$plusargs("DEBUG=%d", debug)) begin end

        if ($value$plusargs("VCD=%s", vcd_path)) begin
            $dumpfile(vcd_path);
            $dumpvars(0, tb_kwt_bias_overlap_task);
        end
        if ($value$plusargs("OUTPUT=%s", output_path)) begin
            output_fd = $fopen(output_path, "w");
            if (output_fd == 0) $fatal(1, "cannot open output file: %s", output_path);
        end
        $readmemh(a_path, a_source);
        $readmemh(w_path, w_source);
        $readmemh(bias_path, bias_source);
        $readmemh(expected_path, expected);
        row_tiles = (m_len + ROWS - 1) / ROWS;
        col_tiles = (n_len + COLS - 1) / COLS;
        total_tiles = row_tiles * col_tiles;
        if (a_preload_arg < 0 || a_preload_arg > row_tiles)
            a_preload_arg = row_tiles;
        a_preload_tiles = a_preload_arg;
        a_preload_words = ((a_preload_tiles > 0) ? a_preload_tiles : 1) * ROWS * k_len;
        if (a_preload_words > m_len * k_len) a_preload_words = m_len * k_len;

        for (int i = 0; i < m_len * k_len; i++) a_store[i] = '0;
        for (int i = 0; i < MAX_K * COLS; i++) begin
            w_page[0][i] = '0;
            w_page[1][i] = '0;
        end

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        task_start_cycle = cycle_count;
        bus_cycles = 0;

        forever begin
            @(negedge clk);

            // Default single-cycle strobes.
            bias_wr_en = 1'b0;
            stage_start_valid = 1'b0;
            out_accept = 1'b0;

            // A/W/bias job starters. These can request in this same cycle.
            if (!w_active && w_group < col_tiles && w_page_is_free(w_group % 2)) begin
                w_active = 1'b1;
                w_page_group[w_group % 2] = w_group;
                w_page_valid[w_group % 2] = 1'b0;
                w_idx = 0;
            end
            if (!bias_active && bias_group < col_tiles) begin
                if (bias_group <= (compute_started ? compute_col + 1 : 0)) begin
                    bias_active = 1'b1;
                    bias_idx = 0;
                    bias_half = 1'b0;
                end
            end

            bus_enable = (bus_gap <= 1) ? 1'b1 : ((cycle_count % bus_gap) != 0);
            grant = bus_gnt;
            bus_cycles++;
            if (grant == '0) bus_idle_cycles++;

            // Shared-bus slot consumption. One grant = one 16-bit slot.
            if (grant[REQ_BIAS]) begin
                bias_wr_en = 1'b1;
                bias_wr_group = 4'(bias_group);
                bias_wr_idx = 4'(bias_idx);
                bias_wr_half = bias_half;
                bias_wr_data = bias_half ? bias_source[bias_group*COLS+bias_idx][31:16]
                                         : bias_source[bias_group*COLS+bias_idx][15:0];
                bias_load_words++;
                bus_grant_bias++;
                if (bias_half) begin
                    bias_half = 1'b0;
                    if (bias_idx == COLS - 1) begin
                        bias_active = 1'b0;
                        bias_group++;
                    end else bias_idx++;
                end else begin
                    bias_half = 1'b1;
                end
            end
            if (grant[REQ_A]) begin
                a_store[a_loaded] = a_source[a_loaded];
                a_loaded++;
                a_load_words++;
                bus_grant_a++;
            end
            if (grant[REQ_W]) begin
                kk = w_idx / COLS;
                cc = w_idx % COLS;
                if (w_group * COLS + cc < n_len) begin
                    w_page[w_group % 2][w_idx] = w_source[kk * n_len + w_group * COLS + cc];
                    w_load_words++;
                end else begin
                    w_page[w_group % 2][w_idx] = '0;
                end
                bus_grant_w++;
                w_fetch_words++;
                w_idx++;
                if (w_idx == k_len * COLS) begin
                    w_page_valid[w_group % 2] = 1'b1;
                    w_active = 1'b0;
                    w_group++;
                end
            end
            if (stage_out_valid && !out_slot_req)
                out_bp_stall_cycles++;
            if (grant[REQ_OUT]) begin
                bus_grant_out++;
                out_slots++;
                if (out_slot_phase) begin
                    out_slot_phase = 1'b0;
                    out_accept = 1'b1;
                end else begin
                    out_slot_phase = 1'b1;
                end
            end

            // ---------------- compute engine ----------------
            if (!compute_done) begin
                case (comp_phase)
                    0: begin
                        if (comp_tile < total_tiles) begin
                            next_tile = comp_tile;
                            next_ct = next_tile / row_tiles;
                            next_rt = next_tile % row_tiles;
                            if (!compute_started) begin
                                if ((a_loaded >= a_preload_words) &&
                                    a_row_ready(0) && w_page_ready(0)) begin
                                    compute_started = 1'b1;
                                    compute_start_cycle = cycle_count;
                                    initial_load_cycles = cycle_count - task_start_cycle;
                                    comp_ct = 0;
                                    comp_rt = 0;
                                    comp_page = 0;
                                    compute_col = 0;
                                    rm = '0;
                                    for (r = 0; r < ROWS; r++)
                                        rm[r] = (r < m_len);
                                    cacc_busy[0] = 1'b1;
                                    w_page_busy[0] = 1'b1;
                                    cur = 0;
                                    comp_phase = 1;
                                end else begin
                                    if (compute_started) load_stall_cycles++;
                                end
                            end else if (cacc_busy[next_tile % 2]) begin
                                cacc_stall_cycles++;
                            end else if (!a_row_ready(next_rt) || !w_page_ready(next_ct)) begin
                                load_stall_cycles++;
                            end else begin
                                comp_ct = next_ct;
                                comp_rt = next_rt;
                                comp_page = next_tile % 2;
                                compute_col = next_ct;
                                rm = '0;
                                for (r = 0; r < ROWS; r++)
                                    rm[r] = (comp_rt * ROWS + r < m_len);
                                cacc_busy[comp_page] = 1'b1;
                                w_page_busy[next_ct % 2] = 1'b1;
                                cur = comp_page;
                                comp_phase = 1;
                            end
                        end else begin
                            compute_done = 1'b1;
                            compute_done_cycle = cycle_count;
                            compute_wall_cycles = cycle_count - compute_start_cycle;
                        end
                    end
                    1: begin
                        clear_bit = 1'b1;
                        stream_valid = 1'b0;
                        comp_phase = 2;
                    end
                    2: begin
                        clear_bit = 1'b0;
                        comp_phase = 3;
                        comp_row_idx = 0;
                    end
                    3: begin
                        if (comp_row_idx < ROWS) begin
                            if (comp_rt * ROWS + comp_row_idx < m_len) begin
                                a_hold[comp_row_idx] = a_store[(comp_rt * ROWS + comp_row_idx) * k_len];
                                a_local_reads++;
                            end else begin
                                a_hold[comp_row_idx] = '0;
                            end
                            comp_row_idx++;
                        end
                        if (comp_row_idx == ROWS) begin
                            comp_phase = 4;
                            mac_i = 0;
                            gap_pending = 1'b0;
                        end
                    end
                    4: begin
                        kk = mac_i / COLS;
                        cc = mac_i % COLS;
                        if (insert_gaps != 0 && (mac_i % 17 == 0) && !gap_pending) begin
                            stream_valid = 1'b0;
                            gap_pending = 1'b1;
                        end else begin
                            stream_valid = 1'b1;
                            gap_pending = 1'b0;
                            ic = 4'(cc);
                            bw = w_page[comp_ct % 2][mac_i][DATA_W-1:0];
                            w_local_reads++;
                            for (r = 0; r < ROWS; r++)
                                ar[r*DATA_W +: DATA_W] = a_hold[r][DATA_W-1:0];
                            if (cc < ROWS && kk + 1 < k_len) begin
                                if (comp_rt * ROWS + cc < m_len) begin
                                    a_next[cc] = a_store[(comp_rt * ROWS + cc) * k_len + kk + 1];
                                    a_local_reads++;
                                end else begin
                                    a_next[cc] = '0;
                                end
                            end
                            if (cc == COLS - 1 && kk + 1 < k_len)
                                for (r = 0; r < ROWS; r++) a_hold[r] = a_next[r];
                            mac_i++;
                            if (mac_i == k_len * COLS) begin
                                // The last MAC beat must remain asserted for the
                                // upcoming posedge; phase 5 deasserts it next cycle.
                                comp_phase = 5;
                                tail_i = 0;
                            end
                        end
                    end
                    5: begin
                        stream_valid = 1'b0;
                        tail_i++;
                        if (tail_i == 5) begin
                            if (debug) $display("DBG tile_done cycle=%0d tile=%0d page=%0d mac_i=%0d wreads=%0d areads=%0d",
                                                cycle_count, comp_tile, comp_page, mac_i,
                                                w_local_reads, a_local_reads);
                            if (debug && comp_tile == 1)
                                $display("DBG psum1_0_0=%0d psum1_0_1=%0d psum1_1_0=%0d",
                                         $signed(core1.g_psum_bank[0].psum_mem[0]),
                                         $signed(core1.g_psum_bank[0].psum_mem[1]),
                                         $signed(core1.g_psum_bank[1].psum_mem[0]));
                            pending_valid[comp_page] = 1'b1;
                            pending_tile[comp_page] = comp_tile;
                            pending_row[comp_page] = comp_rt * ROWS;
                            pending_col[comp_page] = comp_ct * COLS;
                            pending_ct[comp_page] = comp_ct;
                            comp_tile++;
                            if (comp_rt == row_tiles - 1) begin
                                w_group_done[comp_ct] = 1'b1;
                                w_page_busy[comp_ct % 2] = 1'b0;
                            end
                            comp_phase = 0;
                        end
                    end
                    default: comp_phase = 0;
                endcase
            end

            // ---------------- export engine ----------------
            if (exp_busy) begin
                if (stage_capture_done) begin
                    pending_valid[exp_page] = 1'b0;
                    cacc_busy[exp_page] = 1'b0;
                    exp_busy = 1'b0;
                    exp_tile++;
                end
            end
            if (!exp_busy && exp_tile < total_tiles) begin
                p = exp_tile % 2;
                if (pending_valid[p] && pending_tile[p] == exp_tile) begin
                    if (!bias_group_valid[pending_ct[p]]) begin
                        bias_stall_cycles++;
                    end else if (!stage_start_ready) begin
                        stage_stall_cycles++;
                    end else begin
                        exp_busy = 1'b1;
                        exp_page = p;
                        cap_sel = p[0];
                        if (debug) $display("DBG exp_start cycle=%0d tile=%0d page=%0d row=%0d col=%0d",
                                            cycle_count, exp_tile, p, pending_row[p], pending_col[p]);
                        stage_base_row = 7'(pending_row[p]);
                        stage_base_col = 8'(pending_col[p]);
                        stage_m_len = 7'(m_len);
                        stage_n_len = 9'(n_len);
                        stage_start_valid = 1'b1;
                        if (compute_started && !compute_done)
                            overlapped_captures++;
                    end
                end
            end

            // Termination.
            if (compute_done && exp_tile == total_tiles &&
                checked_outputs == m_len * n_len) begin
                tail_cycles = cycle_count - compute_done_cycle;
                if (a_load_words != m_len * k_len ||
                    w_load_words != k_len * n_len ||
                    bias_load_words != 2 * col_tiles * COLS ||
                    out_write_words != 2 * m_len * n_len ||
                    out_slots != 2 * m_len * n_len ||
                    a_local_reads != m_len * k_len * col_tiles ||
                    w_local_reads != row_tiles * col_tiles * k_len * COLS ||
                    checked_outputs != m_len * n_len)
                    $fatal(1, "bias-overlap architectural access count mismatch: a=%0d/%0d w=%0d/%0d bias=%0d/%0d out=%0d/%0d slots=%0d/%0d ar=%0d/%0d wr=%0d/%0d chk=%0d/%0d",
                           a_load_words, m_len*k_len, w_load_words, k_len*n_len,
                           bias_load_words, 2*col_tiles*COLS, out_write_words, 2*m_len*n_len,
                           out_slots, 2*m_len*n_len, a_local_reads,
                           m_len*k_len*col_tiles, w_local_reads,
                           row_tiles*col_tiles*k_len*COLS, checked_outputs, m_len*n_len);
                if (bus_grant_out != 2 * m_len * n_len ||
                    bus_grant_a != a_load_words ||
                    bus_grant_w != w_fetch_words ||
                    bus_grant_bias != bias_load_words ||
                    bus_grant_a + bus_grant_w + bus_grant_bias + bus_grant_out !=
                    (a_load_words + w_fetch_words + bias_load_words + out_write_words))
                    $fatal(1, "shared-bus grant accounting mismatch: out=%0d a=%0d w=%0d bias=%0d",
                           bus_grant_out, bus_grant_a, bus_grant_w, bus_grant_bias);
                $display("ARCH_METRICS cycles=%0d initial_load_cycles=%0d compute_wall_cycles=%0d tail_cycles=%0d cacc_stall_cycles=%0d load_stall_cycles=%0d bias_stall_cycles=%0d stage_stall_cycles=%0d out_bp_stall_cycles=%0d a_load_words=%0d w_load_words=%0d w_fetch_words=%0d bias_load_words=%0d out_write_words=%0d out_slots=%0d a_local_reads=%0d w_local_reads=%0d checked_outputs=%0d checked_padding=%0d overlapped_output_words=%0d overlapped_captures=%0d bus_grant_out=%0d bus_grant_bias=%0d bus_grant_a=%0d bus_grant_w=%0d bus_idle_cycles=%0d bus_cycles=%0d a_preload_tiles=%0d bias_delay=%0d out_bp=%0d bus_gap=%0d",
                         cycle_count, initial_load_cycles, compute_wall_cycles,
                         tail_cycles, cacc_stall_cycles, load_stall_cycles,
                         bias_stall_cycles, stage_stall_cycles, out_bp_stall_cycles,
                         a_load_words, w_load_words, w_fetch_words, bias_load_words,
                         out_write_words, out_slots, a_local_reads, w_local_reads,
                         checked_outputs, checked_padding, overlapped_output_words,
                         overlapped_captures, bus_grant_out, bus_grant_bias,
                         bus_grant_a, bus_grant_w, bus_idle_cycles, bus_cycles,
                         a_preload_tiles, bias_delay, bp_period, bus_gap);
                $display("PASS bias-overlap task M=%0d K=%0d N=%0d gaps=%0d preload=%0d bias_delay=%0d out_bp=%0d bus_gap=%0d",
                         m_len, k_len, n_len, insert_gaps, a_preload_tiles,
                         bias_delay, bp_period, bus_gap);
                if (output_fd != 0) $fclose(output_fd);
                $finish;
            end
        end
    end
endmodule
