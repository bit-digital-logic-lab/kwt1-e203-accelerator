// Single-layer KWT-1 PostNorm encoder assembly (graduate item 5).
//
//   o  = attn_single_head(X)                      (RTL, copied from grad-attention)
//   a  = requant(o, scale_o -> scale_a)           (RTL, this file)
//   r1,out1 = kwt_postnorm_ffn residual1 + LN1    (RTL, copied from grad-nonlinear)
//   h  = kwt_ffn1_row(out1)                       (RTL, this file)
//   g,f,r2,y = kwt_postnorm_ffn GELU/FFN2/residual2/LN2
//
// The datapath is a *chain of RTL arithmetic*: the testbench only loads X and
// parameters, then reads back y (and the per-stage debug snapshots).  No
// multiply, add, softmax, GELU, LayerNorm or requantisation is evaluated in the
// testbench or on the host.
//
// Row flow: the attention module computes the whole 99x64 attention tensor,
// then the wrapper streams one row at a time through
// residual1 -> LN1 -> FFN1 -> GELU -> FFN2 -> residual2 -> LN2.
//
// Unified load port (ld_sel, ld_addr = {row[6:0],col[6:0]} for the attention
// memories, or a flat index for the FFN memories):
//   0 X(99x64) 1 Wq 2 Wk 3 Wv 4 Wo 5 attn_bias(64) 6 exp_lut(64)
//   7 W1(256x64) 8 b1(256,int32) 9 W2(16384) 10 b2(64,int32)
module kwt_encoder_layer #(
    parameter int T   = 99,
    parameter int D   = 64,
    parameter int HID = 256,
    // requantisation from attention output scale to the postnorm `a` scale:
    //   a = (o * M_A + 2^(Q_A-1)) >>> Q_A ,  M_A = round(scale_o/scale_a * 2^Q_A)
    parameter int M_A = 5981,
    parameter int Q_A = 16
) (
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic [3:0]           ld_sel,
    input  logic [13:0]          ld_addr,
    input  logic signed [31:0]   ld_data,
    input  logic                 ld_we,

    input  logic                 start,
    output logic                 busy,
    output logic                 done,

    output logic                 y_valid,
    input  logic                 y_ready,
    output logic signed [D*16-1:0] y_data,

    // per-stage debug snapshots (for the level-by-level comparison testbench)
    output logic                 dbg_a_valid,
    output logic signed [D*16-1:0] dbg_a,
    output logic                 dbg_r1_valid,
    output logic signed [D*16-1:0] dbg_r1,
    output logic                 dbg_g_valid,
    output logic signed [HID*16-1:0] dbg_g,
    output logic                 dbg_f_valid,
    output logic signed [D*32-1:0] dbg_f,
    output logic                 dbg_r2_valid,
    output logic signed [D*16-1:0] dbg_r2,
    output logic [31:0]          o1_sat_count,
    output logic [31:0]          y_sat_count
);

    localparam int T7 = 7;

    // ------------------------------------------------------------ attention
    logic                attn_start, attn_busy, attn_done;
    logic [2:0]          attn_ld_sel;
    logic [6:0]          attn_ld_row, attn_ld_col;
    logic signed [31:0]  attn_ld_data;
    logic                attn_ld_we;
    logic [2:0]          attn_rd_sel;
    logic [6:0]          attn_rd_row, attn_rd_col;
    logic signed [31:0]  attn_rd_data;

    attn_single_head u_attn (
        .clk, .rst_n,
        .start(attn_start), .busy(attn_busy), .done(attn_done),
        .ld_sel(attn_ld_sel), .ld_row(attn_ld_row), .ld_col(attn_ld_col),
        .ld_data(attn_ld_data), .ld_we(attn_ld_we),
        .rd_sel(attn_rd_sel), .rd_row(attn_rd_row), .rd_col(attn_rd_col),
        .rd_data(attn_rd_data));

    // ------------------------------------------------------------ postnorm
    logic                pn_x_valid, pn_a_valid;
    logic                pn_x_ready, pn_a_ready;
    logic signed [D*16-1:0] pn_x_data, pn_a_data;
    logic                pn_h_valid, pn_h_ready;
    logic signed [31:0]  pn_h_data;
    logic                pn_o1_valid, pn_o1_ready;
    logic signed [D*16-1:0] pn_o1_data;
    logic                pn_y_valid, pn_y_ready;
    logic signed [D*16-1:0] pn_y_data;
    logic                pn_w_we, pn_b_we;
    logic [13:0]         pn_w_addr;
    logic signed [15:0]  pn_w_wdata;
    logic [5:0]          pn_b_addr;
    logic signed [31:0]  pn_b_wdata;
    logic                pn_dbg_r1_valid, pn_dbg_g_valid, pn_dbg_f_valid, pn_dbg_r2_valid;
    logic signed [D*16-1:0]  pn_dbg_r1, pn_dbg_r2;
    logic signed [HID*16-1:0] pn_dbg_g;
    logic signed [D*32-1:0]  pn_dbg_f;

    kwt_postnorm_ffn u_pn (
        .clk, .rst_n,
        .x_valid(pn_x_valid), .x_ready(pn_x_ready), .x_data(pn_x_data),
        .a_valid(pn_a_valid), .a_ready(pn_a_ready), .a_data(pn_a_data),
        .h_valid(pn_h_valid), .h_ready(pn_h_ready), .h_data(pn_h_data),
        .o1_valid(pn_o1_valid), .o1_ready(pn_o1_ready), .o1_data(pn_o1_data),
        .y_valid(pn_y_valid), .y_ready(pn_y_ready), .y_data(pn_y_data),
        .w_we(pn_w_we), .w_addr(pn_w_addr), .w_wdata(pn_w_wdata),
        .b_we(pn_b_we), .b_addr(pn_b_addr), .b_wdata(pn_b_wdata),
        .dbg_r1_valid(pn_dbg_r1_valid), .dbg_r1(pn_dbg_r1),
        .dbg_g_valid(pn_dbg_g_valid), .dbg_g(pn_dbg_g),
        .dbg_f_valid(pn_dbg_f_valid), .dbg_f(pn_dbg_f),
        .dbg_r2_valid(pn_dbg_r2_valid), .dbg_r2(pn_dbg_r2),
        .o1_sat_count(o1_sat_count), .y_sat_count(y_sat_count));

    // ------------------------------------------------------------ FFN1
    logic                f1_w_we, f1_b_we, f1_i_we, f1_start, f1_busy, f1_done;
    logic [13:0]         f1_w_addr;
    logic signed [15:0]  f1_w_wdata;
    logic [7:0]          f1_b_addr;
    logic signed [31:0]  f1_b_wdata;
    logic signed [D*16-1:0] f1_i_wdata;
    logic                f1_h_valid, f1_h_ready;
    logic signed [31:0]  f1_h_data;

    kwt_ffn1_row #(.D(D), .HID(HID)) u_ffn1 (
        .clk, .rst_n,
        .w_we(f1_w_we), .w_addr(f1_w_addr), .w_wdata(f1_w_wdata),
        .b_we(f1_b_we), .b_addr(f1_b_addr), .b_wdata(f1_b_wdata),
        .i_we(f1_i_we), .i_wdata(f1_i_wdata),
        .start(f1_start), .busy(f1_busy), .done(f1_done),
        .h_valid(f1_h_valid), .h_ready(f1_h_ready), .h_data(f1_h_data));

    // ------------------------------------------------------------ load decode
    always_comb begin
        attn_ld_sel  = 3'd0;
        attn_ld_row  = 7'd0;
        attn_ld_col  = 7'd0;
        attn_ld_data = ld_data;
        attn_ld_we   = 1'b0;

        f1_w_we = 1'b0; f1_w_addr = 14'd0; f1_w_wdata = ld_data[15:0];
        f1_b_we = 1'b0; f1_b_addr = 8'd0;  f1_b_wdata = ld_data;
        pn_w_we = 1'b0; pn_w_addr = 14'd0; pn_w_wdata = ld_data[15:0];
        pn_b_we = 1'b0; pn_b_addr = 6'd0;  pn_b_wdata = ld_data;

        if (ld_we) begin
            unique case (ld_sel)
                4'd0, 4'd1, 4'd2, 4'd3, 4'd4: begin
                    attn_ld_we  = 1'b1;
                    attn_ld_sel = ld_sel[2:0];
                    attn_ld_row = ld_addr[13:7];
                    attn_ld_col = ld_addr[6:0];
                end
                4'd5: begin attn_ld_we = 1'b1; attn_ld_sel = 3'd5; attn_ld_col = ld_addr[6:0]; end
                4'd6: begin attn_ld_we = 1'b1; attn_ld_sel = 3'd6; attn_ld_row = ld_addr[6:0]; end
                4'd7:  begin f1_w_we = 1'b1; f1_w_addr = ld_addr; end
                4'd8:  begin f1_b_we = 1'b1; f1_b_addr = ld_addr[7:0]; end
                4'd9:  begin pn_w_we = 1'b1; pn_w_addr = ld_addr; end
                4'd10: begin pn_b_we = 1'b1; pn_b_addr = ld_addr[5:0]; end
                default: ;
            endcase
        end
    end

    // ------------------------------------------------------------ FFN1 <-> postnorm
    assign pn_h_valid  = f1_h_valid;
    assign f1_h_ready  = pn_h_ready;
    assign pn_h_data   = f1_h_data;

    // ------------------------------------------------------------ requant o -> a
    logic signed [15:0] o_buf [0:D-1];
    logic signed [15:0] x_buf [0:D-1];
    logic signed [D*16-1:0] a_pack;
    logic signed [39:0] rq_prod;
    logic signed [39:0] rq_val;

    always_comb begin
        a_pack = '0;
        for (int d = 0; d < D; d++) begin
            rq_prod = o_buf[d] * M_A;
            rq_val  = (rq_prod + (40'sd1 <<< (Q_A-1))) >>> Q_A;
            if (rq_val > 40'sd32767)       a_pack[d*16 +: 16] = 16'sd32767;
            else if (rq_val < -40'sd32768) a_pack[d*16 +: 16] = -16'sd32768;
            else                           a_pack[d*16 +: 16] = rq_val[15:0];
        end
    end

    // ------------------------------------------------------------ row spinner
    typedef enum logic [3:0] {
        L_IDLE, L_ATTN_START, L_ATTN_WAIT,
        L_ROW_X, L_ROW_O, L_ROW_WAIT, L_ROW_DRIVE, L_ROW_AP, L_ROW_O1,
        L_ROW_FF_START, L_ROW_FF_WAIT, L_ROW_Y, L_DONE
    } lstate_e;
    lstate_e lstate;

    logic [6:0] row, dcnt;

    always_comb begin
        attn_start  = 1'b0;
        attn_rd_sel = 3'd0;
        attn_rd_row = row;
        attn_rd_col = dcnt;

        pn_x_valid  = 1'b0;
        pn_a_valid  = 1'b0;
        pn_x_data   = '0;
        pn_a_data   = '0;
        pn_o1_ready = 1'b0;
        pn_y_ready  = 1'b0;

        f1_i_we     = 1'b0;
        f1_i_wdata  = '0;
        f1_start    = 1'b0;

        dbg_a_valid  = 1'b0;
        dbg_a        = '0;
        dbg_r1_valid = pn_dbg_r1_valid;
        dbg_r1       = pn_dbg_r1;
        dbg_g_valid  = pn_dbg_g_valid;
        dbg_g        = pn_dbg_g;
        dbg_f_valid  = pn_dbg_f_valid;
        dbg_f        = pn_dbg_f;
        dbg_r2_valid = pn_dbg_r2_valid;
        dbg_r2       = pn_dbg_r2;

        y_valid = pn_y_valid;
        y_data  = pn_y_data;

        for (int d = 0; d < D; d++) begin
            pn_x_data[d*16 +: 16] = x_buf[d];
            pn_a_data[d*16 +: 16] = a_pack[d*16 +: 16];
        end

        case (lstate)
            L_ATTN_START: attn_start = 1'b1;
            L_ROW_X: begin
                attn_rd_sel = 3'd0;
                attn_rd_row = row;
                attn_rd_col = dcnt;
            end
            L_ROW_O: begin
                attn_rd_sel = 3'd6;
                attn_rd_row = row;
                attn_rd_col = dcnt;
            end
            L_ROW_DRIVE: begin
                pn_x_valid = 1'b1;
                pn_a_valid = 1'b1;
            end
            L_ROW_AP: begin
                dbg_a_valid = 1'b1;
                dbg_a       = a_pack;
            end
            L_ROW_O1: begin
                if (pn_o1_valid) begin
                    f1_i_we     = 1'b1;
                    f1_i_wdata  = pn_o1_data;
                    pn_o1_ready = 1'b1;
                end
            end
            L_ROW_FF_START: f1_start = 1'b1;
            L_ROW_Y: pn_y_ready = y_ready;
            default: ;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lstate <= L_IDLE;
            row    <= 7'd0;
            dcnt   <= 7'd0;
        end else begin
            case (lstate)
                L_IDLE: begin
                    row  <= 7'd0;
                    dcnt <= 7'd0;
                    if (start) lstate <= L_ATTN_START;
                end
                L_ATTN_START: lstate <= L_ATTN_WAIT;
                L_ATTN_WAIT:  if (attn_done) begin row <= 7'd0; dcnt <= 7'd0; lstate <= L_ROW_X; end
                L_ROW_X: begin
                    x_buf[dcnt] <= attn_rd_data[15:0];
                    if (dcnt == T7'(D-1)) begin dcnt <= 7'd0; lstate <= L_ROW_O; end
                    else                        dcnt <= dcnt + 7'd1;
                end
                L_ROW_O: begin
                    o_buf[dcnt] <= attn_rd_data[15:0];
                    if (dcnt == T7'(D-1)) begin dcnt <= 7'd0; lstate <= L_ROW_WAIT; end
                    else                        dcnt <= dcnt + 7'd1;
                end
                L_ROW_WAIT:  lstate <= L_ROW_DRIVE;
                L_ROW_DRIVE: if (!pn_x_ready && !pn_a_ready) lstate <= L_ROW_AP;
                L_ROW_AP:    lstate <= L_ROW_O1;
                L_ROW_O1:    if (pn_o1_valid) lstate <= L_ROW_FF_START;
                L_ROW_FF_START: lstate <= L_ROW_FF_WAIT;
                L_ROW_FF_WAIT: if (f1_done) lstate <= L_ROW_Y;
                L_ROW_Y: begin
                    if (pn_y_valid && y_ready) begin
                        if (row == T7'(T-1)) lstate <= L_DONE;
                        else begin row <= row + 7'd1; dcnt <= 7'd0; lstate <= L_ROW_X; end
                    end
                end
                L_DONE: lstate <= L_IDLE;
                default: lstate <= L_IDLE;
            endcase
        end
    end

    assign busy = (lstate != L_IDLE);
    assign done = (lstate == L_DONE);

endmodule
