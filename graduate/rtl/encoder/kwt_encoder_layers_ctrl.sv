// 12-layer KWT-1 encoder parameter-update / data-movement controller (item 6).
//
// This is the E203-side *sequencer* boundary: it owns when parameters are
// loaded, when the layer input is (re)written, when the compute layer starts,
// and how the previous layer output becomes the next layer input.  It contains
// no arithmetic: it only generates `ld_sel/ld_addr/ld_we` transactions and
// routes data.  Parameter *bytes* come from an external region (`p_rdata`,
// driven in the SoC-like testbench by the E203 command/transport model); the
// layer-output -> next-layer-input movement is done here from the captured y
// stream.
//
// Transaction order per layer (matches the single-layer load map):
//   params: sel1 Wq(4096) sel2 Wk(4096) sel3 Wv(4096) sel4 Wo(4096)
//           sel5 bias(64) sel6 exp_lut(64)
//           sel7 W1(16384) sel8 b1(256) sel9 W2(16384) sel10 b2(64)
//   input : sel0 X(99x64)  (layer0: preloaded externally; layer L>0: y of L-1)
//   start -> wait done -> capture y
// After layer 11 the classification-head boundary (`head_valid`, `head_token`)
// is raised.  The softmax/GELU/LayerNorm/Dense classification head itself is a
// *software* stage (see SOFTWARE_BOUNDARY.md); the hardware boundary is
// `head_token` = encoder output row 0 (class token) plus `head_valid`.
module kwt_encoder_layers_ctrl #(
    parameter int T   = 99,
    parameter int D   = 64,
    parameter int HID = 256,
    parameter int LAYERS = 12
) (
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic                 start,
    output logic                 busy,
    output logic                 done,

    output logic [3:0]           u_ld_sel,
    output logic [13:0]          u_ld_addr,
    output logic signed [31:0]   u_ld_data,
    output logic                 u_ld_we,

    output logic                 enc_start,
    input  logic                 enc_busy,
    input  logic                 enc_done,

    input  logic                 y_valid,
    output logic                 y_ready,
    input  logic signed [D*16-1:0] y_data,

    input  logic signed [31:0]   p_rdata,

    output logic                 head_valid,
    input  logic                 head_ready,
    output logic signed [D*16-1:0] head_token,
    output logic [3:0]           layer_id
);

    localparam int NSEG = 10;

    logic [3:0]  seg;
    logic [14:0] idx;
    logic [3:0]  layer;
    logic [6:0]  xrow, xcol;      // X (pre)load / write-back counters
    logic [6:0]  cr;              // y capture row counter
    logic        run_started;

    function automatic int seg_sel(input int s);
        case (s)
            0: seg_sel = 1;  1: seg_sel = 2;  2: seg_sel = 3;  3: seg_sel = 4;
            4: seg_sel = 5;  5: seg_sel = 6;  6: seg_sel = 7;  7: seg_sel = 8;
            8: seg_sel = 9;  9: seg_sel = 10;
            default: seg_sel = 1;
        endcase
    endfunction

    function automatic int seg_len(input int s);
        case (s)
            0,1,2,3: seg_len = 4096;
            4,5:     seg_len = 64;
            6:       seg_len = 16384;
            7:       seg_len = 256;
            8:       seg_len = 16384;
            9:       seg_len = 64;
            default: seg_len = 1;
        endcase
    endfunction

    logic signed [15:0] x_buf [0:T*D-1];
    logic signed [D*16-1:0] head_tok_r;

    typedef enum logic [3:0] {
        C_IDLE, C_PRE_X, C_PL_SEG, C_WR_X, C_RUN, C_HEAD, C_DONE
    } cstate_e;
    cstate_e cstate;

    // combinational address for the 2-D (row-major, 128-stride) attention
    // memories, derived directly from the segment index (no register lag).
    logic [6:0] prow, pcol;
    logic [13:0] attn_addr;
    assign prow = idx[12:6];
    assign pcol = idx[5:0];
    assign attn_addr = {prow, 7'b0} | {7'b0, pcol};

    always_comb begin
        u_ld_sel  = 4'd0;
        u_ld_addr = 14'd0;
        u_ld_data = '0;
        u_ld_we   = 1'b0;
        enc_start = 1'b0;
        y_ready   = 1'b0;
        head_valid = 1'b0;
        head_token = head_tok_r;
        layer_id   = layer;

        case (cstate)
            C_PRE_X: begin
                u_ld_sel  = 4'd0;
                u_ld_addr = {xrow, 7'b0} | {7'b0, xcol};
                u_ld_data = p_rdata;
                u_ld_we   = 1'b1;
            end
            C_PL_SEG: begin
                u_ld_sel  = seg_sel(seg)[3:0];
                u_ld_data = p_rdata;
                u_ld_we   = 1'b1;
                if (seg <= 3) u_ld_addr = attn_addr;
                else          u_ld_addr = idx[13:0];
            end
            C_WR_X: begin
                u_ld_sel  = 4'd0;
                u_ld_addr = {xrow, 7'b0} | {7'b0, xcol};
                u_ld_data = {{16{x_buf[xrow*D + xcol][15]}}, x_buf[xrow*D + xcol]};
                u_ld_we   = 1'b1;
            end
            C_RUN: begin
                y_ready   = 1'b1;
                enc_start = !run_started;
            end
            C_HEAD: head_valid = 1'b1;
            default: ;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cstate     <= C_IDLE;
            seg        <= 4'd0;
            idx        <= 15'd0;
            xrow       <= 7'd0;
            xcol       <= 7'd0;
            cr         <= 7'd0;
            layer      <= 4'd0;
            run_started<= 1'b0;
            head_tok_r <= '0;
        end else begin
            case (cstate)
                C_IDLE: begin
                    if (start) begin
                        layer <= 4'd0; xrow <= 7'd0; xcol <= 7'd0;
                        cstate <= C_PRE_X;
                    end
                end
                C_PRE_X: begin
                    x_buf[xrow*D + xcol] <= p_rdata[15:0];
                    if (xrow == 7'(T-1) && xcol == 7'(D-1)) begin
                        xrow <= 7'd0; xcol <= 7'd0; seg <= 4'd0; idx <= 15'd0;
                        cstate <= C_PL_SEG;
                    end else if (xcol == 7'(D-1)) begin
                        xcol <= 7'd0; xrow <= xrow + 7'd1;
                    end else begin
                        xcol <= xcol + 7'd1;
                    end
                end
                C_PL_SEG: begin
                    if (idx == 15'(seg_len(seg) - 1)) begin
                        idx <= 15'd0;
                        if (seg == 4'(NSEG-1)) begin
                            seg <= 4'd0; xrow <= 7'd0; xcol <= 7'd0;
                            cstate <= C_WR_X;
                        end else begin
                            seg <= seg + 4'd1;
                        end
                    end else begin
                        idx <= idx + 15'd1;
                    end
                end
                C_WR_X: begin
                    if (xrow == 7'(T-1) && xcol == 7'(D-1)) begin
                        xrow <= 7'd0; xcol <= 7'd0; cr <= 7'd0;
                        run_started <= 1'b0;
                        cstate <= C_RUN;
                    end else if (xcol == 7'(D-1)) begin
                        xcol <= 7'd0; xrow <= xrow + 7'd1;
                    end else begin
                        xcol <= xcol + 7'd1;
                    end
                end
                C_RUN: begin
                    if (!run_started) run_started <= 1'b1;
                    if (y_valid && y_ready) begin
                        for (int d = 0; d < D; d++) x_buf[cr*D + d] <= y_data[d*16 +: 16];
                        if (cr == 7'd0 && layer == 4'(LAYERS-1)) head_tok_r <= y_data;
                        cr <= cr + 7'd1;
                    end
                    if (enc_done) begin
                        if (layer == 4'(LAYERS-1)) begin
                            cstate <= C_HEAD;
                        end else begin
                            layer <= layer + 4'd1;
                            seg   <= 4'd0; idx <= 15'd0;
                            cstate <= C_PL_SEG;
                        end
                    end
                end
                C_HEAD: if (head_ready) cstate <= C_DONE;
                C_DONE: cstate <= C_IDLE;
                default: cstate <= C_IDLE;
            endcase
        end
    end

    assign busy = (cstate != C_IDLE);
    assign done = (cstate == C_DONE);

endmodule
