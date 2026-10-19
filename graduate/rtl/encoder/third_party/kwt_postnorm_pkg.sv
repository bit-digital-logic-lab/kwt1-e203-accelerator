// Arithmetic primitives shared by the KWT-1 PostNorm FFN/GELU/residual/LayerNorm
// prototype.  All helpers are pure combinational functions so that the integer
// reference model (model/kwt_postnorm_ref.py) can reproduce them bit-exactly.
//
// Rounding rule used everywhere in the datapath:
//   mul_sr(a,m,q) = (a*m + 2^(q-1)) >>> q        (arithmetic shift / floor)
// which is "round half up" on the signed value and is identical to
//   (a*m + (1 << (q-1))) >> q   in Python for negative operands too.
package kwt_postnorm_pkg;

    // Signed multiply + rounding right shift, 64-bit result.
    function automatic logic signed [63:0] mul_sr64(input logic signed [63:0] a,
                                                    input logic signed [31:0] m,
                                                    input int unsigned q);
        logic signed [63:0] p;
        logic signed [63:0] rounded;
        logic signed [63:0] out;
        begin
            p       = a * m;                         // 64x32 -> 64-bit context
            rounded = p + (64'sd1 <<< (q - 1));
            out     = rounded >>> q;
            mul_sr64 = out;
        end
    endfunction

    // Floor of the integer square root of a 64-bit unsigned value.
    function automatic logic [63:0] isqrt64(input logic [63:0] v);
        logic [63:0] n;
        logic [63:0] res;
        logic [63:0] bpos;
        begin
            n   = v;
            res = 64'd0;
            bpos = 64'h4000_0000_0000_0000;
            for (int i = 0; i < 32; i++) begin
                if (n >= (res + bpos)) begin
                    n   = n - (res + bpos);
                    res = (res >> 1) + bpos;
                end else begin
                    res = res >> 1;
                end
                bpos = bpos >> 2;
            end
            isqrt64 = res;
        end
    endfunction

    // Symmetric saturation to [-clamp, +clamp].  Returns {sat, value} because
    // Icarus 11 requires every function argument to be an input.
    function automatic logic signed [64:0] sat_sym(input logic signed [63:0] v,
                                                   input logic signed [63:0] clamp);
        begin
            if (v > clamp)
                sat_sym = {1'b1, clamp};
            else if (v < -clamp)
                sat_sym = {1'b1, -clamp};
            else
                sat_sym = {1'b0, v};
        end
    endfunction

endpackage
