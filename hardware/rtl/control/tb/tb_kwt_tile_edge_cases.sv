// Tile-level edge-case checks for the provisional system model.
//
// This test intentionally stays below a full layer: it runs a small sequence
// of 8x16 tiles, so the last-row tile can be checked as 3 valid rows plus 5
// padding rows without repeating the 208-tile/12-layer regression.
//
// The golden matrix is supplied by run_tile_edge_cases.py. That script
// recomputes it from the frozen A/W/bias files and cross-checks the supplied
// expected32 file; the testbench itself never derives an answer from the DUT.
module tb_kwt_tile_edge_cases;
    localparam int EXT_MEM_BYTES = 512 * 1024;
    localparam int A_BASE = 'h00000;
    localparam int B_BASE = 'h04000;
    localparam int BIAS_BASE = 'h0c000;
    localparam int OUT_BASE = 'h10000;
    localparam int TILE_BYTES = 512;
    localparam int MAX_CASES = 64;

    logic clk = 0;
    logic rst_n = 0;
    logic cmd_valid = 0, cmd_ready;
    logic [31:0] cmd_a_addr = 0, cmd_b_addr = 0;
    logic [31:0] cmd_bias_addr = 0, cmd_out_addr = 0;
    logic [6:0] cmd_k_len = 64;
    logic [7:0] cmd_row_mask = 8'hff;
    logic [15:0] cmd_col_mask = 16'hffff;
    logic cmd_relu_en = 0;
    logic rsp_valid, rsp_ready = 0;
    logic [1:0] rsp_status;
    logic bus_valid, bus_write, bus_ready;
    logic [31:0] bus_addr;
    logic [127:0] bus_wdata, bus_rdata;
    logic [15:0] bus_wstrb;
    logic bus_rvalid;

    logic [15:0] a_words [0:13*64*8-1];
    logic [15:0] w_words [0:16*64*16-1];
    logic [31:0] bias_words [0:255];
    logic [31:0] golden [0:MAX_CASES*128-1];
    logic [63:0] cases [0:MAX_CASES-1];

    int case_count = 0;
    int reset_index = -1;
    int reset_seen = 0;
    int checked_tiles = 0, checked_valid = 0, checked_padding = 0;
    int continuous_tiles = 0;
    int first_task_cycle = -1, last_task_cycle = -1;
    int cycle_count = 0;
    int task_cycles = 0;
    logic [127:0] sentinel;

    string a_path, w_path, bias_path, golden_path, cases_path;

    always #5 clk = ~clk;

    kwt_tile_system_model #(.EXT_MEM_BYTES(EXT_MEM_BYTES)) dut (.*);
    kwt_ext_memory_model #(.MEM_BYTES(EXT_MEM_BYTES)) memory (
        .clk(clk), .rst_n(rst_n), .valid(bus_valid), .write(bus_write),
        .ready(bus_ready), .byte_addr(bus_addr), .wdata(bus_wdata),
        .wstrb(bus_wstrb), .rvalid(bus_rvalid), .rdata(bus_rdata)
    );

    always @(posedge clk) begin
        if (rst_n) cycle_count <= cycle_count + 1;
    end

    task automatic issue_case(input int index);
        logic [15:0] tile;
        logic [7:0] rt, ct, row_mask;
        logic [63:0] packed_case;
        packed_case = cases[index];
        tile = packed_case[15:0];
        rt = packed_case[23:16];
        ct = packed_case[31:24];
        row_mask = packed_case[39:32];
        if (tile !== 16'(rt) * 16 + 16'(ct))
            $fatal(1, "case %0d tile=%0d does not match rt=%0d ct=%0d",
                   index, tile, rt, ct);
        cmd_a_addr = A_BASE + 32'(rt) * 1024;
        cmd_b_addr = B_BASE + 32'(ct) * 2048;
        cmd_bias_addr = BIAS_BASE + 32'(ct) * 64;
        cmd_out_addr = OUT_BASE + 32'(tile) * TILE_BYTES;
        cmd_k_len = 64;
        cmd_row_mask = row_mask;
        cmd_col_mask = 16'hffff;
        cmd_relu_en = 0;
    endtask

    task automatic send_command;
        @(negedge clk);
        if (!cmd_ready) $fatal(1, "command interface not ready");
        cmd_valid = 1;
        @(negedge clk);
        cmd_valid = 0;
    endtask

    task automatic wait_response(input int index);
        bit got;
        got = 0;
        for (int cycle = 0; cycle < 20000 && !got; cycle++) begin
            if (rsp_valid) begin
                if (rsp_status !== 2'd0)
                    $fatal(1, "case %0d status=%0d expected=0", index, rsp_status);
                rsp_ready = 1;
                @(negedge clk);
                rsp_ready = 0;
                got = 1;
            end else begin
                @(negedge clk);
            end
        end
        if (!got)
            $fatal(1, "case %0d timeout", index);
    endtask

    task automatic check_case(input int index);
        logic [15:0] tile;
        logic [7:0] rt, ct, row_mask;
        int row, col, valid_rows;
        logic [31:0] actual, expected;
        logic [63:0] packed_case;
        packed_case = cases[index];
        tile = packed_case[15:0];
        rt = packed_case[23:16];
        ct = packed_case[31:24];
        row_mask = packed_case[39:32];
        valid_rows = 0;
        for (int r = 0; r < 8; r++) if (row_mask[r]) valid_rows++;
        for (int r = 0; r < 8; r++) begin
            for (int c = 0; c < 16; c++) begin
                row = int'(rt) * 8 + r;
                col = int'(ct) * 16 + c;
                expected = golden[index*128 + r*16 + c];
                actual = memory.mem[(OUT_BASE/16) + int'(tile)*32 + (r*16+c)/4]
                         [((r*16+c)%4)*32 +: 32];
                if (!row_mask[r] && expected !== 32'b0)
                    $fatal(1, "golden padding is nonzero case=%0d [%0d,%0d]",
                           index, row, col);
                if (actual !== expected) begin
                    $display("MISMATCH case=%0d tile=%0d row=%0d col=%0d got=%h golden=%h",
                             index, tile, row, col, actual, expected);
                    $fatal(1, "golden mismatch case=%0d [%0d,%0d]", index, row, col);
                end
                if (row_mask[r]) checked_valid++;
                else checked_padding++;
            end
        end
        $display("CASE %0d tile=%0d rt=%0d ct=%0d mask=%h valid_rows=%0d outputs=%0d padding=%0d",
                 index, tile, rt, ct, row_mask, valid_rows,
                 valid_rows*16, (8-valid_rows)*16);
    endtask

    task automatic run_case(input int index, input bit do_check);
        int start_cycle;
        issue_case(index);
        start_cycle = cycle_count;
        send_command;
        wait_response(index);
        if (do_check) begin
            // Every checked case after the first exercises a new command with
            // no intervening reset, i.e. a continuous-task sequence.
            if (checked_tiles > 0) continuous_tiles++;
            checked_tiles++;
            check_case(index);
        end
        if (first_task_cycle < 0) first_task_cycle = start_cycle;
        last_task_cycle = cycle_count;
        task_cycles += cycle_count - start_cycle;
    endtask

    task automatic reset_in_flight(input int index);
        int start_cycle;
        bit active;
        issue_case(index);
        start_cycle = cycle_count;
        send_command;
        // Hold reset while the accelerator is actively computing, after all
        // operand/bias loads have completed, so partial accumulation state and
        // the output stage are both live.
        active = 0;
        for (int cycle = 0; cycle < 5000 && !active; cycle++) begin
            @(negedge clk);
            if (dut.tile.busy) active = 1;
        end
        if (!active)
            $fatal(1, "reset injection never reached tile compute");
        rst_n = 0;
        repeat (4) @(negedge clk);
        if (rsp_valid)
            $fatal(1, "response asserted during reset injection");
        rst_n = 1;
        repeat (3) @(negedge clk);
        if (!cmd_ready)
            $fatal(1, "command interface not ready after reset");
        reset_seen++;
        $display("RESET case=%0d tile=%0d accepted_cycle=%0d reset_ok=1",
                 index, cases[index][15:0], start_cycle);
    endtask

    initial begin
        int parsed;
        logic [127:0] packed_sentinel;
        if (!$value$plusargs("A=%s", a_path) ||
            !$value$plusargs("W=%s", w_path) ||
            !$value$plusargs("BIAS=%s", bias_path) ||
            !$value$plusargs("GOLDEN=%s", golden_path) ||
            !$value$plusargs("CASES=%s", cases_path))
            $fatal(1, "missing +A/+W/+BIAS/+GOLDEN/+CASES paths");
        parsed = $value$plusargs("CASE_COUNT=%d", case_count);
        parsed = $value$plusargs("RESET_INDEX=%d", reset_index);
        if (case_count < 1 || case_count > MAX_CASES)
            $fatal(1, "invalid CASE_COUNT=%0d", case_count);

        $readmemh(a_path, a_words);
        $readmemh(w_path, w_words);
        $readmemh(bias_path, bias_words);
        $readmemh(golden_path, golden);
        $readmemh(cases_path, cases);

        // Pack the same physical layout as the full-layer testbench.
        for (int rt = 0; rt < 13; rt++)
            for (int k = 0; k < 64; k++)
                for (int r = 0; r < 8; r++)
                    memory.mem[(A_BASE/16) + rt*64 + k][r*16 +: 16] =
                        a_words[(rt*64+k)*8+r];
        for (int ct = 0; ct < 16; ct++) begin
            for (int k = 0; k < 64; k++)
                for (int c = 0; c < 16; c++)
                    memory.mem[(B_BASE/16) + ct*128 + k*2 + c/8][(c%8)*16 +: 16] =
                        w_words[(ct*64+k)*16+c];
            for (int c = 0; c < 16; c++)
                memory.mem[(BIAS_BASE/16) + ct*4 + c/4][(c%4)*32 +: 32] =
                    bias_words[ct*16+c];
        end
        sentinel = 128'h0123_4567_89ab_cdef_fedc_ba98_7654_3210;
        memory.mem[(OUT_BASE/16) + 208*32] = sentinel;

        repeat (3) @(negedge clk);
        rst_n = 1;
        for (int i = 0; i < case_count; i++) begin
            if (i == reset_index) reset_in_flight(i);
            else run_case(i, 1'b1);
        end
        if (reset_index >= 0 && !reset_seen)
            $fatal(1, "reset injection did not complete");
        packed_sentinel = memory.mem[(OUT_BASE/16) + 208*32];
        if (packed_sentinel !== sentinel)
            $fatal(1, "write escaped the 208-tile output region");
        $display("PASS tile edge cases: cases=%0d checked_tiles=%0d valid=%0d padding=%0d reset=%0d continuous=%0d task_cycles=%0d total_cycles=%0d",
                 case_count, checked_tiles, checked_valid, checked_padding,
                 reset_seen, continuous_tiles,
                 last_task_cycle-first_task_cycle+1, cycle_count);
        $finish;
    end
endmodule
