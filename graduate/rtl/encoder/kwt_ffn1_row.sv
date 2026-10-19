// FFN1 row engine for the graduate single-layer PostNorm encoder assembly.
//
// Computes one encoder row of the first feed-forward projection
//     h[c] = sum_j o1[j] * W1[c][j] + b1_int[c]      c in [0,HID), j in [0,D)
// where
//   o1    : LayerNorm1 output row, INT16, scale sx (A12W10 frozen activation scale)
//   W1    : INT16 A12W10 weights, per-output-channel scale s_w1[c]
//   b1[c] = round(bias1[c] / (sx * s_w1[c]))       (frozen pack integer bias)
// so that h is in the per-channel scale s_h[c] = sx * s_w1[c] that the
// kwt_postnorm_ffn `h` port consumes.  This is the SAME integer vector the
// existing undergraduate FFN1 core produces; here it is a single-row engine so
// the layer wrapper can stream it directly behind LayerNorm1.
//
// All arithmetic is performed here (RTL), never in the testbench.  Accumulator
// is INT48; |h|max <= D * 2047 * 1023 ~ 1.34e8, far inside INT32/INT48.
//
// Weight layout matches the frozen A12W10 pack
// (`software/results/ffn1_a12w10_static_sym.npz`, layer01_weight[D][HID] in
// [k][n] order): w_addr = c*D + j.
module kwt_ffn1_row #(
    parameter int D   = 64,
    parameter int HID = 256
) (
    input  logic                clk,
    input  logic                rst_n,

    input  logic                w_we,
    input  logic [13:0]         w_addr,
    input  logic signed [15:0]  w_wdata,
    input  logic                b_we,
    input  logic [7:0]          b_addr,
    input  logic signed [31:0]  b_wdata,
    input  logic                i_we,
    input  logic signed [D*16-1:0] i_wdata,

    input  logic                start,
    output logic                busy,
    output logic                done,

    output logic                h_valid,
    input  logic                h_ready,
    output logic signed [31:0]  h_data
);

    logic signed [15:0] w_mem [0:HID*D-1];
    logic signed [31:0] b_mem [0:HID-1];
    logic signed [15:0] i_mem [0:D-1];

    logic signed [47:0] acc;
    logic [7:0]  c;
    logic [5:0]  j;
    logic        running;

    typedef enum logic [1:0] { S_IDLE, S_MAC, S_OUT } state_e;
    state_e state;

    always_ff @(posedge clk) begin
        if (w_we) w_mem[w_addr] <= w_wdata;
        if (b_we) b_mem[b_addr] <= b_wdata;
        if (i_we) begin
            for (int t = 0; t < D; t++) i_mem[t] <= i_wdata[t*16 +: 16];
        end
    end

    always_comb begin
        h_valid = (state == S_OUT);
        h_data  = acc[31:0] + b_mem[c];
        busy    = running;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            running <= 1'b0;
            done    <= 1'b0;
            acc     <= 48'sd0;
            c       <= 8'd0;
            j       <= 6'd0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    running <= 1'b0;
                    if (start) begin
                        acc     <= 48'sd0;
                        c       <= 8'd0;
                        j       <= 6'd0;
                        running <= 1'b1;
                        state   <= S_MAC;
                    end
                end
                S_MAC: begin
                    acc <= acc + (i_mem[j] * w_mem[c*D + j]);
                    if (j == D[5:0] - 6'd1) state <= S_OUT;
                    else                    j     <= j + 6'd1;
                end
                S_OUT: begin
                    if (h_ready) begin
                        acc <= 48'sd0;
                        j   <= 6'd0;
                        if (c == HID[7:0] - 8'd1) begin
                            running <= 1'b0;
                            done    <= 1'b1;
                            state   <= S_IDLE;
                        end else begin
                            c     <= c + 8'd1;
                            state <= S_MAC;
                        end
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
