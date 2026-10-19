// KWT-1 PostNorm single-layer encoder fragment: GELU, FFN2 (256->64), the two
// residual additions and the two LayerNorms.
//
// Official PostNorm order (software/run.py / transformer_utils.py, prenorm=False):
//
//   attn_out          = attention(X)
//   r1                = X + attn_out                 -> dbg_r1
//   out1              = layernorm1(r1)               -> o1_data  (fed to FFN1)
//   h                 = out1 @ W1 + b1               (external FFN1 hardware)
//   g                 = gelu(h)                      -> dbg_g
//   f                 = g @ W2 + b2                  -> dbg_f
//   r2                = out1 + f                     -> dbg_r2
//   y                 = layernorm2(r2)               -> y_data   (encoder output)
//
// The FFN1 stage is NOT part of this file: `h` enters as an INT32 stream so the
// block drops straight behind the existing `kwt_row_broadcast_tile` / output
// stage, one hidden channel per beat, and `out1` leaves through a valid/ready
// port that the FFN1 loader can consume.
//
// Quantization scales are frozen in `kwt_postnorm_cfg.svh` (generated):
//   s_x  : layer input X            s_a : attention output
//   sx   : LN1 output / FFN1 input (== frozen A12W10 activation scale)
//   s_h[c] = sx * s_w1[c]           s_g : GELU output / FFN2 input
//   s_h2[k] = s_g * s_w2[k]         s_r1, s_r2 : residual scales
//   s_out: encoder output
// All datapath rounding is `mul_sr` round-half-up; activations saturate to the
// A12 container [-2047, +2047].
//
// One row is processed at a time (no cross-row pipelining); every interface is
// valid/ready and stalls correctly under backpressure.
module kwt_postnorm_ffn (
    input  logic clk,
    input  logic rst_n,

    // layer input row X, per-tensor scale s_x
    input  logic x_valid,
    output logic x_ready,
    input  logic signed [D*16-1:0] x_data,
    // attention output row, per-tensor scale s_a
    input  logic a_valid,
    output logic a_ready,
    input  logic signed [D*16-1:0] a_data,
    // FFN1 hidden stream, scale s_h[c]; c = index of the accepted beat (0..HID-1)
    input  logic h_valid,
    output logic h_ready,
    input  logic signed [31:0] h_data,
    // LayerNorm1 output (scale sx), forwarded to the FFN1 loader
    output logic o1_valid,
    input  logic o1_ready,
    output logic signed [D*16-1:0] o1_data,
    // encoder output row (scale s_out)
    output logic y_valid,
    input  logic y_ready,
    output logic signed [D*16-1:0] y_data,

    // FFN2 INT16 weights / INT32 bias load port (behavioural SRAM)
    input  logic w_we,
    input  logic [$clog2(HID*D)-1:0] w_addr,
    input  logic signed [15:0] w_wdata,
    input  logic b_we,
    input  logic [$clog2(D)-1:0] b_addr,
    input  logic signed [31:0] b_wdata,

    // per-stage snapshots for the bit-exact testbench
    output logic dbg_r1_valid,
    output logic signed [D*16-1:0] dbg_r1,
    output logic dbg_g_valid,
    output logic signed [HID*16-1:0] dbg_g,
    output logic dbg_f_valid,
    output logic signed [D*32-1:0] dbg_f,
    output logic dbg_r2_valid,
    output logic signed [D*16-1:0] dbg_r2,
    output logic [31:0] o1_sat_count,
    output logic [31:0] y_sat_count
);
    `include "kwt_postnorm_cfg.svh"

    localparam int ACT_W = 16;

    // ---- coefficient vectors exposed by the generated configuration ----
    // Packed rather than unpacked arrays: the target Yosys build does not
    // accept unpacked-array ports.
    logic signed [HID*32-1:0] mg_w;
    logic signed [D*32-1:0]   mb1_w, md2_w, g1f_w, b1f_w, g2f_w, b2f_w;
    for (genvar gi = 0; gi < HID; gi++) begin : g_mg
        assign mg_w[gi*32 +: 32] = MG_F(gi);
    end
    for (genvar gi = 0; gi < D; gi++) begin : g_coef
        assign mb1_w[gi*32 +: 32] = MB1;
        assign md2_w[gi*32 +: 32] = MD2_F(gi);
        assign g1f_w[gi*32 +: 32] = G1F_F(gi);
        assign b1f_w[gi*32 +: 32] = B1F_F(gi);
        assign g2f_w[gi*32 +: 32] = G2F_F(gi);
        assign b2f_w[gi*32 +: 32] = B2F_F(gi);
    end

    // ---- stage 1: residual + LayerNorm1 ----
    logic res1_in_valid, res1_in_ready, res1_out_valid, res1_out_ready, res1_sat;
    logic ln1_in_valid, ln1_in_ready, ln1_out_valid, ln1_out_ready, ln1_sat;
    logic signed [D*ACT_W-1:0] res1_data, ln1_data;
    logic signed [D*ACT_W-1:0] r1_lat, out1_lat;
    // ---- stage 2: GELU + FFN2 + residual + LayerNorm2 ----
    logic ffn2_start, ffn2_busy, ffn2_done;
    logic signed [D*32-1:0] ffn2_f;
    logic res2_in_valid, res2_in_ready, res2_out_valid, res2_out_ready, res2_sat;
    logic ln2_in_valid, ln2_in_ready, ln2_out_valid, ln2_out_ready, ln2_sat;
    logic signed [D*ACT_W-1:0] res2_data, ln2_data, r2_lat, y_lat;
    logic signed [HID*ACT_W-1:0] g_vec, g_vec_next;
    logic signed [15:0] gelu_g;
    logic [8:0] hc;

    always_comb begin
        g_vec_next = g_vec;
        g_vec_next[hc*ACT_W +: ACT_W] = gelu_g;
    end

    kwt_residual_add #(.D(D), .A_W(16), .B_W(16), .OUT_W(ACT_W), .CLAMP(CLAMP_A), .QB(QB),
                       .MA_W(32), .MB_W(32)) u_res1 (
        .clk, .rst_n,
        .in_valid(res1_in_valid), .in_ready(res1_in_ready),
        .a_data(x_data), .b_data(a_data),
        .ma(MA1), .mb(mb1_w),
        .out_valid(res1_out_valid), .out_ready(res1_out_ready),
        .out_data(res1_data), .out_sat(res1_sat));

    kwt_layernorm #(.D(D), .IN_W(16), .OUT_W(ACT_W), .CLAMP(CLAMP_A), .ACC_W(48),
                    .GAMMA_FRAC(GAMMA_FRAC), .GOUT_FRAC(GOUT_FRAC),
                    .ISQ_FRAC(ISQ_FRAC), .VAR_FRAC(VAR_FRAC), .ISQ_MAX(ISQ_MAX)) u_ln1 (
        .clk, .rst_n,
        .in_valid(ln1_in_valid), .in_ready(ln1_in_ready),
        .x_data(res1_data), .eps_q(EPS_Q1), .gcoef(g1f_w), .bcoef(b1f_w), .mo(MO1),
        .out_valid(ln1_out_valid), .out_ready(ln1_out_ready),
        .out_data(ln1_data), .out_sat(ln1_sat));

    kwt_gelu u_gelu (.h_int(h_data), .mg(mg_w[hc*32 +: 32]), .mgn(MGN), .g_int(gelu_g));

    kwt_ffn2 #(.HID(HID), .OUT(D), .IN_W(ACT_W), .W_W(16), .ACC_W(48), .OUT_W(32)) u_ffn2 (
        .clk, .rst_n,
        .w_we(w_we), .w_addr(w_addr), .w_wdata(w_wdata),
        .b_we(b_we), .b_addr(b_addr), .b_wdata(b_wdata),
        .start(ffn2_start), .g_data(g_vec), .busy(ffn2_busy), .done(ffn2_done),
        .f_data(ffn2_f));

    kwt_residual_add #(.D(D), .A_W(16), .B_W(32), .OUT_W(ACT_W), .CLAMP(CLAMP_A), .QB(QB),
                       .MA_W(32), .MB_W(32)) u_res2 (
        .clk, .rst_n,
        .in_valid(res2_in_valid), .in_ready(res2_in_ready),
        .a_data(out1_lat), .b_data(ffn2_f),
        .ma(MC2), .mb(md2_w),
        .out_valid(res2_out_valid), .out_ready(res2_out_ready),
        .out_data(res2_data), .out_sat(res2_sat));

    kwt_layernorm #(.D(D), .IN_W(16), .OUT_W(ACT_W), .CLAMP(CLAMP_A), .ACC_W(48),
                    .GAMMA_FRAC(GAMMA_FRAC), .GOUT_FRAC(GOUT_FRAC),
                    .ISQ_FRAC(ISQ_FRAC), .VAR_FRAC(VAR_FRAC), .ISQ_MAX(ISQ_MAX)) u_ln2 (
        .clk, .rst_n,
        .in_valid(ln2_in_valid), .in_ready(ln2_in_ready),
        .x_data(res2_data), .eps_q(EPS_Q2), .gcoef(g2f_w), .bcoef(b2f_w), .mo(MO2),
        .out_valid(ln2_out_valid), .out_ready(ln2_out_ready),
        .out_data(ln2_data), .out_sat(ln2_sat));

    // ---- sequencer ----
    typedef enum logic [3:0] {
        S_IDLE, S_R1, S_R1W, S_LN1, S_LN1W, S_O1,
        S_H, S_FF, S_FFW, S_R2, S_R2W, S_LN2, S_LN2W, S_Y
    } state_e;
    state_e state;

    always_comb begin
        res1_in_valid = 1'b0;
        res1_out_ready = 1'b0;
        ln1_in_valid = 1'b0;
        ln1_out_ready = 1'b0;
        res2_in_valid = 1'b0;
        res2_out_ready = 1'b0;
        ln2_in_valid = 1'b0;
        ln2_out_ready = 1'b0;
        ffn2_start = 1'b0;
        x_ready = 1'b0;
        a_ready = 1'b0;
        h_ready = 1'b0;
        o1_valid = 1'b0;
        o1_data = '0;
        y_valid = 1'b0;
        y_data = '0;
        dbg_r1_valid = 1'b0;
        dbg_g_valid = 1'b0;
        dbg_f_valid = 1'b0;
        dbg_r2_valid = 1'b0;
        case (state)
            S_IDLE: begin
                x_ready = 1'b1;
                a_ready = 1'b1;
            end
            S_R1:   res1_in_valid = 1'b1;
            S_R1W:  res1_out_ready = 1'b1;
            S_LN1:  ln1_in_valid = 1'b1;
            S_LN1W: begin
                ln1_out_ready = 1'b1;
                dbg_r1_valid = 1'b1;
            end
            S_O1: begin
                o1_valid = 1'b1;
                o1_data = out1_lat;
            end
            S_H:    h_ready = 1'b1;
            S_FF: begin
                ffn2_start = 1'b1;
                dbg_g_valid = 1'b1;
            end
            S_R2:   res2_in_valid = 1'b1;
            S_R2W: begin
                res2_out_ready = 1'b1;
                dbg_f_valid = 1'b1;
            end
            S_LN2:  ln2_in_valid = 1'b1;
            S_LN2W: begin
                ln2_out_ready = 1'b1;
                dbg_r2_valid = 1'b1;
            end
            S_Y: begin
                y_valid = 1'b1;
                y_data = y_lat;
            end
            default: ;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            r1_lat        <= '0;
            out1_lat      <= '0;
            r2_lat        <= '0;
            y_lat         <= '0;
            g_vec         <= '0;
            hc            <= 9'd0;
            dbg_r1        <= '0;
            dbg_g         <= '0;
            dbg_f         <= '0;
            dbg_r2        <= '0;
            o1_sat_count  <= 32'd0;
            y_sat_count   <= 32'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (x_valid && a_valid) state <= S_R1;
                end
                S_R1: begin
                    if (res1_in_ready) state <= S_R1W;
                end
                S_R1W: begin
                    if (res1_out_valid) begin
                        r1_lat <= res1_data;
                        dbg_r1 <= res1_data;
                        state  <= S_LN1;
                    end
                end
                S_LN1: begin
                    if (ln1_in_ready) state <= S_LN1W;
                end
                S_LN1W: begin
                    if (ln1_out_valid) begin
                        out1_lat <= ln1_data;
                        if (ln1_sat) o1_sat_count <= o1_sat_count + 32'd1;
                        state    <= S_O1;
                    end
                end
                S_O1: begin
                    if (o1_ready) begin
                        hc    <= 9'd0;
                        state <= S_H;
                    end
                end
                S_H: begin
                    if (h_valid && h_ready) begin
                        g_vec <= g_vec_next;
                        if (hc == HID[8:0] - 9'd1) begin
                            dbg_g <= g_vec_next;
                            state <= S_FF;
                        end else begin
                            hc <= hc + 9'd1;
                        end
                    end
                end
                S_FF: begin
                    state <= S_FFW;
                end
                S_FFW: begin
                    if (ffn2_done) begin
                        dbg_f <= ffn2_f;
                        state <= S_R2;
                    end
                end
                S_R2: begin
                    if (res2_in_ready) state <= S_R2W;
                end
                S_R2W: begin
                    if (res2_out_valid) begin
                        r2_lat <= res2_data;
                        dbg_r2 <= res2_data;
                        state  <= S_LN2;
                    end
                end
                S_LN2: begin
                    if (ln2_in_ready) state <= S_LN2W;
                end
                S_LN2W: begin
                    if (ln2_out_valid) begin
                        y_lat  <= ln2_data;
                        if (ln2_sat) y_sat_count <= y_sat_count + 32'd1;
                        state  <= S_Y;
                    end
                end
                S_Y: begin
                    if (y_ready) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
