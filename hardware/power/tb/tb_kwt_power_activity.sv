`include "e203_defines.v"
`define SHARED cpu.u_e203_cpu.u_e203_nice_core.shared_ram

// Activity/task-cycle evidence testbench for the frozen A12W10 8x16 E203 path.
//
// This is a power-activity variant of
// hardware/rtl/control/tb/tb_kwt_real_layer_e203.sv. It runs the exact same
// frozen real-sample operands through the same NICE-like system model, but it
// also dumps a bounded VCD scope so hardware activity can be inspected.
//
// It is NOT a power model: it emits switching activity for the modeled logic
// only. It does not represent the competition SoC, the unannotated circuits,
// any SRAM macro, or a PDK power/energy number.
module tb_kwt_power_activity;
    localparam int M = 99, K = 64, N = 256;
    localparam int ROW_TILES = 13, COL_TILES = 16, TILES = 208;
    localparam int DESC_BASE = 'h1000;
    localparam int A_BASE = 'h3000;
    localparam int B_BASE = 'h7000;
    localparam int BIAS_BASE = 'hf000;
    localparam int OUT_BASE = 'h10000;
    logic clk = 0, rst_n = 0;
    wire [`E203_PC_SIZE-1:0] inspect_pc;
    reg [7:0] program_bytes [0:63];
    reg [8*512-1:0] program_path;
    reg [8*512-1:0] vcd_path;
    string a_path, w_path, bias_path, expected_path;
    reg [15:0] a_words [0:ROW_TILES*K*8-1];
    reg [15:0] w_words [0:COL_TILES*K*16-1];
    reg [31:0] bias_words [0:N-1];
    reg [31:0] expected_words [0:M*N-1];
    int cycle_count = 0;
    int first_req_cycle = -1, last_rsp_cycle = -1;
    int nice_requests = 0, nice_responses = 0;
    int checked_outputs = 0, checked_padding = 0;
    int run_start = 0, run_tiles = TILES;
    // Window markers, one cycle wide, aligned with the task-cycle definition
    // used by the sibling functional testbench (first NICE request ->
    // last NICE response).
    logic marker_first_req = 0, marker_last_rsp = 0;

    always #5 clk = ~clk;
    `include "e203_cpu_instance.svh"

    task automatic check_tile(input integer tile);
        int rt, ct;
        rt = tile / COL_TILES;
        ct = tile % COL_TILES;
        for (int r = 0; r < 8; r++) begin
            for (int c = 0; c < 16; c++) begin
                int row, col;
                int signed expected;
                logic [31:0] actual;
                row = rt*8+r;
                col = ct*16+c;
                expected = 0;
                if (row < M) begin
                    expected = $signed(expected_words[row*N+col]);
                    checked_outputs++;
                end else checked_padding++;
                actual = `SHARED.mem[(OUT_BASE/16)+tile*32+(r*16+c)/4]
                         [((r*16+c)%4)*32 +: 32];
                if (actual !== 32'(expected))
                    $fatal(1, "layer output [%0d,%0d] got=%h expected=%h",
                           row,col,actual,32'(expected));
            end
        end
    endtask

    always @(posedge clk) begin
        if (rst_n) cycle_count <= cycle_count + 1;
        marker_first_req <= 1'b0;
        marker_last_rsp <= 1'b0;
        if (rst_n && cpu.u_e203_cpu.u_e203_nice_core.nice_req_valid &&
            cpu.u_e203_cpu.u_e203_nice_core.nice_req_ready) begin
            if (cpu.u_e203_cpu.u_e203_nice_core.nice_req_rs1 !==
                32'(DESC_BASE + (run_start+nice_requests)*32))
                $fatal(1, "descriptor pointer mismatch at tile %0d", nice_requests);
            if (cpu.u_e203_cpu.u_e203_nice_core.nice_req_inst[14:12] !== 3'b110)
                $fatal(1, "wrong custom instruction");
            if (nice_requests == 0) begin
                first_req_cycle <= cycle_count;
                marker_first_req <= 1'b1;
                $display("E203 power activity: first request at cycle %0d", cycle_count);
            end
            nice_requests <= nice_requests + 1;
        end
        if (rst_n && cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_valid &&
            cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_ready) begin
            if (cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_err ||
                cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_rdat !== 32'b0)
                $fatal(1, "tile %0d returned error %h", nice_responses,
                       cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_rdat);
            nice_responses <= nice_responses + 1;
            last_rsp_cycle <= cycle_count;
            if ((nice_responses + 1) == run_tiles)
                marker_last_rsp <= 1'b1;
        end
    end

    initial begin
        int parsed;
        logic vcd_on;
        parsed = $value$plusargs("TILE_START=%d", run_start);
        parsed = $value$plusargs("TILE_COUNT=%d", run_tiles);
        if (run_start < 0 || run_tiles < 1 || run_start + run_tiles > TILES)
            $fatal(1, "invalid tile range start=%0d count=%0d", run_start, run_tiles);
        if (!$value$plusargs("PROGRAM=%s", program_path))
            $fatal(1, "missing +PROGRAM=byte-hex-file");
        if (!$value$plusargs("A=%s", a_path) ||
            !$value$plusargs("W=%s", w_path) ||
            !$value$plusargs("BIAS=%s", bias_path) ||
            !$value$plusargs("EXPECTED=%s", expected_path))
            $fatal(1, "missing +A/+W/+BIAS/+EXPECTED hex paths");
        vcd_on = $value$plusargs("VCD=%s", vcd_path);
        if (vcd_on) begin
            $dumpfile(vcd_path);
            // Bounded scopes only. Big behavioral SRAM arrays are deliberately
            // excluded so the VCD stays analyzable; this is a coverage
            // boundary recorded in the report.
            $dumpvars(0, clk);
            $dumpvars(0, rst_n);
            $dumpvars(0, cycle_count);
            $dumpvars(0, first_req_cycle);
            $dumpvars(0, last_rsp_cycle);
            $dumpvars(0, nice_requests);
            $dumpvars(0, nice_responses);
            $dumpvars(0, run_start);
            $dumpvars(0, run_tiles);
            $dumpvars(0, marker_first_req);
            $dumpvars(0, marker_last_rsp);
            // E203 NICE wrapper (kwt_nice_system_model port level).
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_req_valid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_req_ready);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_req_rs1);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_req_inst);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_valid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_ready);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_rdat);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_err);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.bus_valid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.bus_write);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.bus_ready);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.bus_rvalid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.bus_addr);
            // Accelerator control FSM (A12W10 8x16 tile system).
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.state);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.status_q);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_cmd_ready);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_rsp_valid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_rsp_status);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.desc_bus_owner);
            // Tile system: load / compute / writeback phases.
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.state);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile_busy);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile_done);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.dma_busy);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.dma_done);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.dma_error);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.k_len_q);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.bias_index);
            // DMA engine.
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.dma.state);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.dma.remaining);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.dma.lane);
            // Compute engine + product array + cacc activity.
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.state);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.busy);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.done);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.page_q);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.k_index);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.lane_index);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.drain_count);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.write_index);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.product_valid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.cacc_clear);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.dump_req);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.compute_mem_en);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.a_bus);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.b_bus);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.array.launch);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.array.product_valid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.cacc.dump_req);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.cacc.dump_valid);
            $dumpvars(0, cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.tile.cacc.collision_seen);
        end
        $readmemh(a_path, a_words);
        $readmemh(w_path, w_words);
        $readmemh(bias_path, bias_words);
        $readmemh(expected_path, expected_words);
        for (int i = 0; i < 64; i++) program_bytes[i] = 0;
        $readmemh(program_path, program_bytes);
        for (int word = 0; word < 8; word++)
            for (int byte_i = 0; byte_i < 8; byte_i++)
                cpu.u_e203_srams.u_e203_itcm_ram.u_e203_itcm_gnrl_ram.u_sirv_sim_ram.mem_r[word][byte_i*8 +: 8] =
                    program_bytes[word*8+byte_i];

        for (int rt = 0; rt < ROW_TILES; rt++)
            for (int k = 0; k < K; k++)
                for (int r = 0; r < 8; r++)
                    `SHARED.mem[(A_BASE/16) + rt*64 + k][r*16 +: 16] =
                        a_words[(rt*K+k)*8+r];
        for (int ct = 0; ct < COL_TILES; ct++) begin
            for (int k = 0; k < K; k++)
                for (int c = 0; c < 16; c++)
                    `SHARED.mem[(B_BASE/16) + ct*128 + k*2 + c/8][(c%8)*16 +: 16] =
                        w_words[(ct*K+k)*16+c];
            for (int c = 0; c < 16; c++)
                `SHARED.mem[(BIAS_BASE/16) + ct*4 + c/4][(c%4)*32 +: 32] =
                    bias_words[ct*16+c];
        end
        for (int rt = 0; rt < ROW_TILES; rt++) begin
            for (int ct = 0; ct < COL_TILES; ct++) begin
                int tile;
                int row_mask;
                tile = rt*COL_TILES + ct;
                row_mask = (rt == ROW_TILES-1) ? 7 : 255;
                `SHARED.mem[(DESC_BASE/16) + tile*2] = {
                    32'(OUT_BASE + tile*512),
                    32'(BIAS_BASE + ct*64),
                    32'(B_BASE + ct*2048),
                    32'(A_BASE + rt*1024)
                };
                `SHARED.mem[(DESC_BASE/16) + tile*2 + 1] = {
                    64'b0, 8'b0, 16'hffff, 8'(row_mask), 32'd64
                };
            end
        end
        if (`SHARED.mem[DESC_BASE/16] !== {32'(OUT_BASE),32'(BIAS_BASE),32'(B_BASE),32'(A_BASE)})
            $fatal(1, "descriptor table initialization failed");

        repeat (10) @(negedge clk);
        rst_n = 1;
        for (int cycle = 0; cycle < 2000000; cycle++) begin
            @(negedge clk);
            if (cpu.u_e203_cpu.u_e203_core.u_e203_exu.u_e203_exu_regfile.rf_r[13] === 32'd2)
                $fatal(1, "CPU detected tile failure after %0d responses", nice_responses);
            if (cpu.u_e203_cpu.u_e203_core.u_e203_exu.u_e203_exu_regfile.rf_r[13] === 32'd1) begin
                if (nice_requests != run_tiles || nice_responses != run_tiles)
                    $fatal(1, "request/response count %0d/%0d expected %0d",
                           nice_requests, nice_responses, run_tiles);
                for (int tile = run_start; tile < run_start+run_tiles; tile++)
                    check_tile(tile);
                $display("PASS frozen real-sample E203 power-activity tiles: start=%0d count=%0d outputs=%0d padding=%0d task_cycles=%0d total_cycles=%0d first_req_cycle=%0d last_rsp_cycle=%0d",
                         run_start, nice_responses, checked_outputs, checked_padding,
                         last_rsp_cycle-first_req_cycle+1, cycle_count,
                         first_req_cycle, last_rsp_cycle);
                $finish;
            end
        end
        $fatal(1, "full FFN1 layer timeout PC=%h requests=%0d responses=%0d",
               inspect_pc, nice_requests, nice_responses);
    end
endmodule

`undef SHARED
