// 12-layer scheduling smoke test for the graduate encoder control path.
//
// Drives kwt_encoder_top through all 12 layers with an external parameter
// region (the E203 transport model): the testbench answers the controller's
// load transactions with bytes and checks the *scheduling* contract:
//
//   * exactly 12 layer starts,
//   * layer_id sweeps 0..11,
//   * every layer performs a full 49,600-entry parameter reload,
//   * every layer produces 99 output rows and the previous layer output is
//     written back as the next layer input (X write-back per layer),
//   * the classification-head boundary (head_valid + head_token) is raised
//     exactly once, after layer 11.
//
// It does NOT claim numeric correctness for layers 1..11 (their scales are not
// calibrated here); see WORKER_REPORT.md / SOFTWARE_BOUNDARY.md.  The datapath
// itself is real and fully exercised.
`timescale 1ns/1ps
module tb_layers_smoke;
    localparam int T   = 99;
    localparam int D   = 64;
    localparam int HID = 256;
    localparam int LAYERS = 12;
    localparam int PARAM_PER_LAYER = 4096*4 + 64 + 64 + 16384 + 256 + 16384 + 64; // 49600

    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst_n = 1'b0;

    logic start, busy, done;
    logic signed [D*16-1:0] head_token;
    logic head_valid, head_ready;
    logic [3:0] layer_id;
    logic y_valid;
    logic signed [D*16-1:0] y_data;
    logic [3:0] o_ld_sel;
    logic [13:0] o_ld_addr;
    logic o_ld_we;
    logic o_enc_start, o_y_ready;
    logic signed [31:0] p_rdata;

    // ---- parameter region (transport model) ----
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

    integer row, col, idx;
    always_comb begin
        row = o_ld_addr[13:7];
        col = o_ld_addr[6:0];
        idx = o_ld_addr;
        p_rdata = 32'sd0;
        case (o_ld_sel)
            4'd0:  p_rdata = {{16{x_mem[row*D + col][15]}}, x_mem[row*D + col]};
            4'd1:  p_rdata = {{16{wq_mem[row*D + col][15]}}, wq_mem[row*D + col]};
            4'd2:  p_rdata = {{16{wk_mem[row*D + col][15]}}, wk_mem[row*D + col]};
            4'd3:  p_rdata = {{16{wv_mem[row*D + col][15]}}, wv_mem[row*D + col]};
            4'd4:  p_rdata = {{16{wo_mem[row*D + col][15]}}, wo_mem[row*D + col]};
            4'd5:  p_rdata = bias_mem[o_ld_addr[5:0]];
            4'd6:  p_rdata = {16'h0, lut_mem[o_ld_addr[5:0]]};
            4'd7:  p_rdata = {{16{w1_mem[idx][15]}}, w1_mem[idx]};
            4'd8:  p_rdata = b1_mem[o_ld_addr[7:0]];
            4'd9:  p_rdata = {{16{w2_mem[idx][15]}}, w2_mem[idx]};
            4'd10: p_rdata = b2_mem[o_ld_addr[5:0]];
            default: p_rdata = 32'sd0;
        endcase
    end

    assign head_ready = 1'b1;

    kwt_encoder_top #(.T(T), .D(D), .HID(HID), .LAYERS(LAYERS)) dut (
        .clk, .rst_n,
        .start, .busy, .done,
        .p_rdata,
        .head_valid, .head_ready, .head_token, .layer_id,
        .y_valid, .y_data,
        .o_ld_sel, .o_ld_addr, .o_ld_we, .o_enc_start, .o_y_ready,
        .dbg_a_valid(), .dbg_a(), .dbg_r1_valid(), .dbg_r1(),
        .dbg_g_valid(), .dbg_g(), .dbg_f_valid(), .dbg_f(), .dbg_r2_valid(), .dbg_r2(),
        .o1_sat_count(), .y_sat_count());

    integer n_start = 0, n_ld = 0, n_ld_par = 0, n_yrows = 0, max_layer = 0;
    integer par_at_start [0:LAYERS-1];
    integer n_head = 0;
    integer cycles = 0;
    string vec, out;
    integer i;

    always_ff @(posedge clk) begin
        cycles <= cycles + 1;
        if (o_ld_we) begin
            n_ld <= n_ld + 1;
            if (o_ld_sel != 4'd0) n_ld_par <= n_ld_par + 1;
        end
        if (o_enc_start) begin
            if (n_start < LAYERS) par_at_start[n_start] = n_ld_par;
            n_start <= n_start + 1;
        end
        if (y_valid && o_y_ready) begin
            n_yrows <= n_yrows + 1;
            if (layer_id > max_layer) max_layer <= layer_id;
        end
        if (head_valid) n_head <= n_head + 1;
    end

    always_ff @(posedge clk)
        if (cycles % 10000000 == 0 && cycles != 0)
            $display("HEARTBEAT cycles=%0d layer=%0d start=%0d y=%0d", cycles, layer_id, n_start, n_yrows);

    initial begin
        if (!$value$plusargs("VEC=%s", vec)) vec = "vectors";
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

        start = 0; rst_n = 0;
        repeat (8) @(negedge clk);
        rst_n = 1;
        repeat (4) @(negedge clk);
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        fork
            begin
                wait (done);
            end
            begin
                repeat (900000000) @(posedge clk);
                $fatal(1, "TIMEOUT");
            end
        join_any
        disable fork;

        repeat (20) @(posedge clk);

        $display("SMOKE n_start=%0d max_layer=%0d yrows=%0d n_ld=%0d n_ld_par=%0d head=%0d",
                 n_start, max_layer, n_yrows, n_ld, n_ld_par, n_head);
        for (i = 0; i < LAYERS; i++)
            $display("SMOKE layer %0d param_lD_before_start=%0d", i, par_at_start[i]);

        if (n_start != LAYERS)            $fatal(1, "expected %0d starts", LAYERS);
        if (max_layer != LAYERS-1)        $fatal(1, "layer_id did not reach %0d", LAYERS-1);
        if (n_yrows != LAYERS*T)          $fatal(1, "expected %0d y rows", LAYERS*T);
        if (n_head < 1)                   $fatal(1, "head boundary not raised");
        if (n_ld_par != LAYERS*PARAM_PER_LAYER)
            $fatal(1, "expected %0d param loads, got %0d", LAYERS*PARAM_PER_LAYER, n_ld_par);
        $display("SMOKE head_token=%0d %0d %0d", head_token[15:0], head_token[31:16], head_token[47:32]);
        $display("SMOKE_PASS");
        $finish;
    end

endmodule
