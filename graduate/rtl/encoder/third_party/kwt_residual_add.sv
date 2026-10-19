// Residual add with per-lane scale alignment and symmetric saturation.
//
//   out[i] = sat( mul_sr(a[i], ma, QB) + mul_sr(b[i], mb[i], QB), [-CLAMP, +CLAMP] )
//
// `a` is a per-tensor quantized vector with scale s_a, `b` is a vector whose
// lanes may carry different scales s_b[i] (per-output-channel weight scales).
// The coefficients ma = round(s_a/s_r * 2^QB) and mb[i] = round(s_b[i]/s_r * 2^QB)
// are calibrated offline so both operands land in the common residual scale s_r.
// A_W and B_W may differ (residual 1 adds two INT16 vectors, residual 2 adds an
// INT16 vector and the INT32 FFN2 output).
//
// Single-entry valid/ready stage: the result is held until out_ready.
module kwt_residual_add #(
    parameter int D     = 64,
    parameter int A_W   = 16,
    parameter int B_W   = 16,
    parameter int OUT_W = 16,
    parameter int CLAMP = 2047,
    parameter int QB    = 16,
    parameter int MA_W  = 32,
    parameter int MB_W  = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [D*A_W-1:0] a_data,
    input  logic signed [D*B_W-1:0] b_data,
    input  logic signed [MA_W-1:0]  ma,
    input  logic signed [D*MB_W-1:0] mb,
    output logic out_valid,
    input  logic out_ready,
    output logic signed [D*OUT_W-1:0] out_data,
    output logic out_sat
);
    logic signed [63:0] comb [0:D-1];
    logic               comb_sat;
    logic signed [64:0] sat_r;
    logic signed [63:0] sa, sb, sum;

    always_comb begin
        comb_sat = 1'b0;
        for (int i = 0; i < D; i++) begin
            sa = kwt_postnorm_pkg::mul_sr64($signed(a_data[i*A_W +: A_W]), ma, QB);
            sb = kwt_postnorm_pkg::mul_sr64($signed(b_data[i*B_W +: B_W]), mb[i*MB_W +: MB_W], QB);
            sum  = sa + sb;
            sat_r = kwt_postnorm_pkg::sat_sym(sum, 64'(CLAMP));
            comb[i] = sat_r[63:0];
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
