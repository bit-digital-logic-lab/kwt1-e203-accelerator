// GELU activation for the FFN1 hidden vector, in the integer domain.
//
// The FFN1 output h_int is in scale s_h[c] = s_x * s_w1[c] (per hidden channel).
// This block
//   1. rescales  h_int  into the shared nonlinear domain, Q(GELU_FRAC) real units:
//        v = round(h_int * s_h[c] * 2^GELU_FRAC) = mul_sr(h_int, MG[c], QG)
//      and clamps v to [-8, +7.996...] in that domain;
//   2. evaluates gelu(v / 2^GELU_FRAC) with a 257-node piecewise-linear LUT
//      (nodes every 1/16, linear interpolation between them);
//   3. rescales the Q8 result into the FFN2 activation scale s_g:
//        g_int = round(gelu * 2^GELU_FRAC * MGN / 2^QGN) = mul_sr(g8, MGN, QGN)
//      and clamps to the A12 activation container.
//
// Purely combinational; the enclosing pipeline registers the result.
module kwt_gelu (
    input  logic signed [31:0] h_int,
    input  logic signed [31:0] mg,
    input  logic signed [31:0] mgn,
    output logic signed [15:0] g_int
);
    `include "kwt_postnorm_cfg.svh"

    localparam int GELU_IN_MAX = (1 << (GELU_FRAC + 3)) - 1;  // +8 - 1/256 in Q8
    localparam int GELU_IN_MIN = -(1 << (GELU_FRAC + 3));     // -8 in Q8
    localparam int IDX_BIAS    = 1 << (GELU_FRAC + 3);        // +2048
    localparam int FRAC_BITS   = 4;
    localparam int GELU_BIAS_A = -CLAMP_A;

    logic signed [15:0] lut_w [0:LUT_N-1];
    logic signed [63:0] p, vq, g8w, res64;
    logic signed [31:0] v, l0, l1, interp;
    logic        [12:0] idx;
    logic        [8:0]  li_idx;
    logic        [3:0]  frac;

    for (genvar i = 0; i < LUT_N; i++)
        assign lut_w[i] = GELU_LUT_F(i);

    always_comb begin
        p  = $signed(h_int) * $signed(mg);
        vq = (p + (64'sd1 <<< (QG - 1))) >>> QG;
        if (vq > 64'(GELU_IN_MAX))
            v = GELU_IN_MAX;
        else if (vq < 64'(GELU_IN_MIN))
            v = GELU_IN_MIN;
        else
            v = vq[31:0];

        idx    = 13'(v + 32'(IDX_BIAS));
        li_idx = idx[12:FRAC_BITS];
        frac   = idx[FRAC_BITS-1:0];

        l0     = $signed(lut_w[li_idx]);
        l1     = $signed(lut_w[li_idx + 1]);
        interp = (l1 - l0) * $signed({1'b0, frac});
        g8w    = 64'(l0) + 64'(interp >>> FRAC_BITS);

        res64 = kwt_postnorm_pkg::mul_sr64(g8w, mgn, QGN);
        if (res64 > 64'(CLAMP_A))
            g_int = 16'(CLAMP_A);
        else if (res64 < -64'(CLAMP_A))
            g_int = 16'(GELU_BIAS_A);
        else
            g_int = res64[15:0];
    end
endmodule
