// Integer LayerNorm over the last dimension (D = 64 = d_model).
//
//   y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
//
// Population variance (mean of squared deviations), matching
// tf.keras.layers.LayerNormalization(epsilon=1e-6) as used by KWT-1.
//
// Everything is integer.  `x[i]` carries the input scale s_ln, so
// var_q = mean((x-mean)^2) is in units of s_ln^2 and
// eps_q = round(eps / s_ln^2 * 2^VAR_FRAC).  The reciprocal standard deviation
//
//   isq = round(2^ISQ_FRAC / sqrt(var + eps))
//
// is obtained with a combinational integer square root plus one integer
// division, then folded together with the precomputed coefficients
//   gcoef[i] = round(gamma[i] * 2^GAMMA_FRAC)
//   mo       = round(2^GOUT_FRAC / (2^GAMMA_FRAC * s_out))
//   bcoef[i] = round(beta[i] / s_out * 2^GOUT_FRAC)
// The degenerate case var == 0 and eps_q == 0 saturates isq to ISQ_MAX so no
// division by zero can occur; the final result is symmetrically clamped.
module kwt_layernorm #(
    parameter int D          = 64,
    parameter int IN_W       = 16,
    parameter int OUT_W      = 16,
    parameter int CLAMP      = 2047,
    parameter int ACC_W      = 48,
    parameter int GAMMA_FRAC = 16,
    parameter int GOUT_FRAC  = 16,
    parameter int ISQ_FRAC   = 16,
    parameter int VAR_FRAC   = 16,
    parameter int ISQ_MAX    = (1 << 24) - 1
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [D*IN_W-1:0]  x_data,
    input  logic signed [ACC_W-1:0]   eps_q,
    input  logic signed [32*D-1:0]    gcoef,
    input  logic signed [32*D-1:0]    bcoef,
    input  logic signed [31:0]        mo,
    output logic out_valid,
    input  logic out_ready,
    output logic signed [D*OUT_W-1:0] out_data,
    output logic out_sat
);
    localparam int SHIFT = $clog2(D);
    localparam int HALF_VAR_FRAC = VAR_FRAC / 2;

    logic signed [63:0] comb [0:D-1];
    logic               comb_sat;
    logic signed [63:0] sum_q, mean, d, t, u, v;
    logic        [63:0] ss_q, var_q, vq, sq, num, isq;
    logic signed [64:0] sat_r;

    always_comb begin
        sum_q = 64'sd0;
        for (int i = 0; i < D; i++)
            sum_q += $signed(x_data[i*IN_W +: IN_W]);
        mean = sum_q >>> SHIFT;

        ss_q = 64'd0;
        for (int i = 0; i < D; i++) begin
            d = $signed(x_data[i*IN_W +: IN_W]) - mean;
            ss_q += d * d;
        end
        var_q = ss_q >>> SHIFT;
        vq = (var_q <<< VAR_FRAC) + {{16{eps_q[ACC_W-1]}}, eps_q};
        sq = kwt_postnorm_pkg::isqrt64(vq);

        num = 64'd1 <<< (ISQ_FRAC + HALF_VAR_FRAC);
        if (sq == 64'd0)
            isq = 64'(ISQ_MAX);
        else begin
            isq = (num + (sq >> 1)) / sq;
            if (isq > 64'(ISQ_MAX))
                isq = 64'(ISQ_MAX);
        end

        comb_sat = 1'b0;
        for (int i = 0; i < D; i++) begin
            d = $signed(x_data[i*IN_W +: IN_W]) - mean;
            t = d * $signed(gcoef[i*32 +: 32]);
            u = kwt_postnorm_pkg::mul_sr64(t, $signed(isq[31:0]), ISQ_FRAC);
            v = kwt_postnorm_pkg::mul_sr64(u, mo, GOUT_FRAC) + $signed(bcoef[i*32 +: 32]);
            sat_r = kwt_postnorm_pkg::sat_sym(v, 64'(CLAMP));
            comb[i]  = sat_r[63:0];
            comb_sat |= sat_r[64];
        end
    end

    assign in_ready = ~out_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_sat   <= 1'b0;
            out_data  <= '0;
        end else if (in_valid && in_ready) begin
            for (int i = 0; i < D; i++)
                out_data[i*OUT_W +: OUT_W] <= comb[i][OUT_W-1:0];
            out_sat   <= comb_sat;
            out_valid <= 1'b1;
        end else if (out_valid && out_ready) begin
            out_valid <= 1'b0;
        end
    end
endmodule
