// One-layer RTL testbench for the encoder_12numeric 12-layer numerical chain.
//
// This is the task's OWN testbench (the assembled prototype's tb_encoder.sv is
// left untouched).  One binary is built per encoder layer with a per-layer
// include directory holding that layer's `attn_params.svh` (attention shifts)
// and `kwt_postnorm_cfg.svh` (residual/LayerNorm/GELU/FFN2 coefficients).  The
// requantisation coefficient M_A is a top-level parameter (Verilator -G).
//
// The testbench performs NO arithmetic: it only loads bytes over the unified
// load port and records the RTL dumps.  Bit-exactness and FP32 error are
// decided in Python (model/run_rtl_chain.py).
`timescale 1ns/1ps
module tb_chain_layer #(
    parameter int T   = 99,
    parameter int D   = 64,
    parameter int HID = 256,
    parameter int M_A = 5981,
    parameter int Q_A = 16
);
    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst_n = 1'b0;

    logic [3:0]        ld_sel;
    logic [13:0]       ld_addr;
    logic signed [31:0] ld_data;
    logic              ld_we;
    logic              start, busy, done;

    logic              y_valid;
    logic              y_ready;
    logic signed [D*16-1:0] y_data;

    logic              dbg_a_valid, dbg_r1_valid, dbg_g_valid, dbg_f_valid, dbg_r2_valid;
    logic signed [D*16-1:0]   dbg_a, dbg_r1, dbg_r2;
    logic signed [HID*16-1:0] dbg_g;
    logic signed [D*32-1:0]   dbg_f;
    logic [31:0] o1_sat_count, y_sat_count;

    assign y_ready = 1'b1;

    kwt_encoder_layer #(.T(T), .D(D), .HID(HID), .M_A(M_A), .Q_A(Q_A)) dut (
        .clk, .rst_n,
        .ld_sel, .ld_addr, .ld_data, .ld_we,
        .start, .busy, .done,
        .y_valid, .y_ready, .y_data,
        .dbg_a_valid, .dbg_a,
        .dbg_r1_valid, .dbg_r1,
        .dbg_g_valid, .dbg_g,
        .dbg_f_valid, .dbg_f,
        .dbg_r2_valid, .dbg_r2,
        .o1_sat_count, .y_sat_count);

    logic [15:0] x_mem   [0:T*D-1];
    logic [15:0] wq_mem  [0:D*D-1];
    logic [15:0] wk_mem  [0:D*D-1];
    logic [15:0] wv_mem  [0:D*D-1];
    logic [15:0] wo_mem  [0:D*D-1];
    logic [31:0] bias_mem[0:D-1];
    logic [15:0] lut_mem [0:D-1];
    logic [15:0] w1_mem  [0:HID*D-1];
    logic [31:0] b1_mem  [0:HID-1];
    logic [15:0] w2_mem  [0:D*HID-1];
    logic [31:0] b2_mem  [0:D-1];

    integer y_fd, a_fd, r1_fd, g_fd, f_fd, r2_fd;
    integer y_cnt = 0, a_cnt = 0, r1_cnt = 0, g_cnt = 0, f_cnt = 0, r2_cnt = 0;
    integer cycles = 0;
    string vec, out;
    integer i, r, c;

    always_ff @(posedge clk) cycles <= cycles + 1;
    always_ff @(posedge clk)
        if (cycles % 2000000 == 0 && cycles != 0)
            $display("HEARTBEAT cycles=%0d rows=%0d", cycles, y_cnt);

    always_ff @(posedge clk) begin
        if (y_valid && y_ready) begin
            if ($isunknown(y_data)) $fatal(1, "y contains X");
            for (int k = 0; k < D; k++) $fwrite(y_fd, "%04x", y_data[k*16 +: 16]);
            $fwrite(y_fd, "\n"); y_cnt <= y_cnt + 1;
        end
        if (dbg_a_valid) begin
            for (int k = 0; k < D; k++) $fwrite(a_fd, "%04x", dbg_a[k*16 +: 16]);
            $fwrite(a_fd, "\n"); a_cnt <= a_cnt + 1;
        end
        if (dbg_r1_valid) begin
            for (int k = 0; k < D; k++) $fwrite(r1_fd, "%04x", dbg_r1[k*16 +: 16]);
            $fwrite(r1_fd, "\n"); r1_cnt <= r1_cnt + 1;
        end
        if (dbg_g_valid) begin
            for (int k = 0; k < HID; k++) $fwrite(g_fd, "%04x", dbg_g[k*16 +: 16]);
            $fwrite(g_fd, "\n"); g_cnt <= g_cnt + 1;
        end
        if (dbg_f_valid) begin
            for (int k = 0; k < D; k++) $fwrite(f_fd, "%08x", dbg_f[k*32 +: 32]);
            $fwrite(f_fd, "\n"); f_cnt <= f_cnt + 1;
        end
        if (dbg_r2_valid) begin
            for (int k = 0; k < D; k++) $fwrite(r2_fd, "%04x", dbg_r2[k*16 +: 16]);
            $fwrite(r2_fd, "\n"); r2_cnt <= r2_cnt + 1;
        end
    end

    task automatic load_lin(input [3:0] sel, input [13:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            ld_we = 1'b1; ld_sel = sel; ld_addr = addr; ld_data = data;
        end
    endtask

    initial begin
        if (!$value$plusargs("VEC=%s", vec)) vec = "vectors";
        if (!$value$plusargs("OUT=%s", out)) out = "build";
        $readmemh({vec, "/sel0_x.hex"},   x_mem);
        $readmemh({vec, "/sel1_wq.hex"},  wq_mem);
        $readmemh({vec, "/sel2_wk.hex"},  wk_mem);
        $readmemh({vec, "/sel3_wv.hex"},  wv_mem);
        $readmemh({vec, "/sel4_wo.hex"},  wo_mem);
        $readmemh({vec, "/sel5_bias.hex"},bias_mem);
        $readmemh({vec, "/sel6_lut.hex"}, lut_mem);
        $readmemh({vec, "/sel7_w1.hex"},  w1_mem);
        $readmemh({vec, "/sel8_b1.hex"},  b1_mem);
        $readmemh({vec, "/sel9_w2.hex"},  w2_mem);
        $readmemh({vec, "/sel10_b2.hex"}, b2_mem);

        y_fd  = $fopen({out, "/y.hex"}, "w");
        a_fd  = $fopen({out, "/a.hex"}, "w");
        r1_fd = $fopen({out, "/r1.hex"}, "w");
        g_fd  = $fopen({out, "/g.hex"}, "w");
        f_fd  = $fopen({out, "/f.hex"}, "w");
        r2_fd = $fopen({out, "/r2.hex"}, "w");

        ld_we = 1'b0; ld_sel = 0; ld_addr = 0; ld_data = 0; start = 0;
        rst_n = 1'b0;
        repeat (8) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        for (i = 0; i < T*D; i++) begin
            r = i / D; c = i % D;
            load_lin(4'd0, {r[6:0], 7'b0} | {7'b0, c[6:0]},
                     {{16{x_mem[i][15]}}, x_mem[i]});
        end
        for (i = 0; i < D*D; i++) begin
            r = i / D; c = i % D;
            load_lin(4'd1, {r[6:0], 7'b0} | {7'b0, c[6:0]}, {{16{wq_mem[i][15]}}, wq_mem[i]});
        end
        for (i = 0; i < D*D; i++) begin
            r = i / D; c = i % D;
            load_lin(4'd2, {r[6:0], 7'b0} | {7'b0, c[6:0]}, {{16{wk_mem[i][15]}}, wk_mem[i]});
        end
        for (i = 0; i < D*D; i++) begin
            r = i / D; c = i % D;
            load_lin(4'd3, {r[6:0], 7'b0} | {7'b0, c[6:0]}, {{16{wv_mem[i][15]}}, wv_mem[i]});
        end
        for (i = 0; i < D*D; i++) begin
            r = i / D; c = i % D;
            load_lin(4'd4, {r[6:0], 7'b0} | {7'b0, c[6:0]}, {{16{wo_mem[i][15]}}, wo_mem[i]});
        end
        for (i = 0; i < D; i++) load_lin(4'd5, i[13:0], bias_mem[i]);
        for (i = 0; i < D; i++) load_lin(4'd6, i[13:0], {16'h0, lut_mem[i]});
        for (i = 0; i < HID*D; i++) load_lin(4'd7, i[13:0], {{16{w1_mem[i][15]}}, w1_mem[i]});
        for (i = 0; i < HID; i++)   load_lin(4'd8, i[13:0], b1_mem[i]);
        for (i = 0; i < D*HID; i++) load_lin(4'd9, i[13:0], {{16{w2_mem[i][15]}}, w2_mem[i]});
        for (i = 0; i < D; i++)     load_lin(4'd10, i[13:0], b2_mem[i]);

        @(negedge clk);
        ld_we = 1'b0;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        fork
            begin
                wait (done);
            end
            begin
                repeat (40000000) @(posedge clk);
                $fatal(1, "TIMEOUT");
            end
        join_any
        disable fork;

        repeat (20) @(posedge clk);
        $display("TB_DONE cycles=%0d y=%0d a=%0d r1=%0d g=%0d f=%0d r2=%0d sat=%0d/%0d",
                 cycles, y_cnt, a_cnt, r1_cnt, g_cnt, f_cnt, r2_cnt,
                 o1_sat_count, y_sat_count);
        $fclose(y_fd); $fclose(a_fd); $fclose(r1_fd); $fclose(g_fd); $fclose(f_fd); $fclose(r2_fd);
        if (y_cnt != T) $fatal(1, "expected %0d y rows, got %0d", T, y_cnt);
        $display("TB_PASS");
        $finish;
    end

endmodule
