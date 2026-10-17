`include "e203_defines.v"
`define SHARED cpu.u_e203_cpu.u_e203_nice_core.shared_ram

module tb_kwt_full_layer_e203;
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
    int cycle_count = 0;
    int first_req_cycle = -1, last_rsp_cycle = -1;
    int nice_requests = 0, nice_responses = 0;
    int checked_outputs = 0, checked_padding = 0;
    int run_start = 0, run_tiles = TILES;

    always #5 clk = ~clk;
    `include "e203_cpu_instance.svh"

    function automatic integer a_value(input integer row, input integer k);
        return ((row*3 + k*5) % 15) - 7;
    endfunction
    function automatic integer b_value(input integer k, input integer col);
        return ((k*7 + col*2) % 13) - 6;
    endfunction
    function automatic integer bias_value(input integer col);
        return (col % 17) - 8;
    endfunction

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
                    expected = bias_value(col);
                    for (int k = 0; k < K; k++)
                        expected += a_value(row,k)*b_value(k,col);
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
        if (rst_n && cpu.u_e203_cpu.u_e203_nice_core.nice_req_valid &&
            cpu.u_e203_cpu.u_e203_nice_core.nice_req_ready) begin
            if (cpu.u_e203_cpu.u_e203_nice_core.nice_req_rs1 !==
                32'(DESC_BASE + (run_start+nice_requests)*32))
                $fatal(1, "descriptor pointer mismatch at tile %0d", nice_requests);
            if (cpu.u_e203_cpu.u_e203_nice_core.nice_req_inst[14:12] !== 3'b110)
                $fatal(1, "wrong custom instruction");
            if (nice_requests == 0) begin
                first_req_cycle <= cycle_count;
                $display("E203 full layer: first request at cycle %0d", cycle_count);
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
            if (nice_responses < 4)
                $display("E203 full layer: tile %0d completed at cycle %0d",
                         nice_responses, cycle_count);
            if ((nice_responses + 1) % 32 == 0)
                $display("E203 full layer: %0d/%0d selected tiles completed", nice_responses + 1, run_tiles);
        end
    end

    initial begin
        int parsed;
        parsed = $value$plusargs("TILE_START=%d", run_start);
        parsed = $value$plusargs("TILE_COUNT=%d", run_tiles);
        if (run_start < 0 || run_tiles < 1 || run_start + run_tiles > TILES)
            $fatal(1, "invalid tile range start=%0d count=%0d", run_start, run_tiles);
        if (!$value$plusargs("PROGRAM=%s", program_path))
            $fatal(1, "missing +PROGRAM=byte-hex-file");
        for (int i = 0; i < 64; i++) program_bytes[i] = 0;
        $readmemh(program_path, program_bytes);
        for (int word = 0; word < 8; word++)
            for (int byte_i = 0; byte_i < 8; byte_i++)
                cpu.u_e203_srams.u_e203_itcm_ram.u_e203_itcm_gnrl_ram.u_sirv_sim_ram.mem_r[word][byte_i*8 +: 8] =
                    program_bytes[word*8+byte_i];

        // Prepacked A row tiles, B column tiles, and biases. Each source tile
        // is reused by many commands, so all data and 208 descriptors fit in
        // the 512 KiB simulation RAM at once.
        for (int rt = 0; rt < ROW_TILES; rt++)
            for (int k = 0; k < K; k++)
                for (int r = 0; r < 8; r++)
                    `SHARED.mem[(A_BASE/16) + rt*64 + k][r*16 +: 16] =
                        16'((rt*8+r < M) ? a_value(rt*8+r,k) : 0);
        for (int ct = 0; ct < COL_TILES; ct++) begin
            for (int k = 0; k < K; k++)
                for (int c = 0; c < 16; c++)
                    `SHARED.mem[(B_BASE/16) + ct*128 + k*2 + c/8][(c%8)*16 +: 16] =
                        16'(b_value(k,ct*16+c));
            for (int c = 0; c < 16; c++)
                `SHARED.mem[(BIAS_BASE/16) + ct*4 + c/4][(c%4)*32 +: 32] =
                    32'(bias_value(ct*16+c));
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
                $display("PASS real E203 FFN1 tiles: start=%0d count=%0d outputs=%0d padding=%0d task_cycles=%0d total_cycles=%0d",
                         run_start, nice_responses, checked_outputs, checked_padding,
                         last_rsp_cycle-first_req_cycle+1, cycle_count);
                $finish;
            end
        end
        $fatal(1, "full FFN1 layer timeout PC=%h requests=%0d responses=%0d nice_state=%0d tile_state=%0d bus_valid=%b bus_rvalid=%b",
               inspect_pc, nice_requests, nice_responses,
               cpu.u_e203_cpu.u_e203_nice_core.accelerator.state,
               cpu.u_e203_cpu.u_e203_nice_core.accelerator.tile_system.state,
               cpu.u_e203_cpu.u_e203_nice_core.bus_valid,
               cpu.u_e203_cpu.u_e203_nice_core.bus_rvalid);
    end
endmodule

`undef SHARED
