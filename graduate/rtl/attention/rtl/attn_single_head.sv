// Single-head self-attention prototype for KWT-1 (layer-0 attention), synthesizable RTL.
//
// Scope (task item 4): Q/K/V 64->64 projections, QK^T / sqrt(64) with the 99x99
// score matrix, row-wise softmax, probability-weighted V (context), output
// projection. All arithmetic is integer fixed-point and is computed here, never
// in the testbench or host. It is bit-for-bit equal to model/attn_int_model.py.
//
// Numeric contract: activations/weights are int16 with symmetric zero point 0
// and power-of-two re-quantization shifts. Softmax uses a 64-entry LUT for
// 2^(f/64) plus a per-row reciprocal divider. See graduate/rtl/attention/README.md.
//
// Dataflow / reuse: X is read once per output column (64 uses per row); Q row i
// is reused across all 99 score columns; K rows are streamed as the transposed
// operand (K[j] dotted with Q[i] gives S[i][j] without materialising K^T); V rows
// are reused across every query row in the context phase; C rows are reused
// across the 64 output columns. Nothing is recomputed from host memory.
//
// Debug/verify read port: rd_sel {0:x,1:q,2:k,3:v,4:p,5:c,6:o,7:bias}. This read
// port is a plain synchronous-memory-style access used only for verification; it
// does not perform arithmetic.

`include "attn_params.svh"

module attn_single_head (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    output logic               busy,
    output logic               done,

    // load port (input / parameter writeback)
    input  logic [2:0]         ld_sel,
    input  logic [6:0]         ld_row,
    input  logic [6:0]         ld_col,
    input  logic signed [31:0] ld_data,
    input  logic               ld_we,

    // verify read port
    input  logic [2:0]         rd_sel,
    input  logic [6:0]         rd_row,
    input  logic [6:0]         rd_col,
    output logic signed [31:0] rd_data
);

    // ---------------------------------------------------------------- memories
    logic signed [15:0] x_mem [0:ATT_T-1][0:ATT_D-1];
    logic signed [15:0] wq    [0:ATT_D-1][0:ATT_D-1];
    logic signed [15:0] wk    [0:ATT_D-1][0:ATT_D-1];
    logic signed [15:0] wv    [0:ATT_D-1][0:ATT_D-1];
    logic signed [15:0] wo    [0:ATT_D-1][0:ATT_D-1];
    logic signed [31:0] bias_m[0:ATT_D-1];
    logic [15:0]        exp_lut[0:(1<<F_BITS)-1];

    logic signed [15:0] q_mem [0:ATT_T-1][0:ATT_D-1];
    logic signed [15:0] k_mem [0:ATT_T-1][0:ATT_D-1];
    logic signed [15:0] v_mem [0:ATT_T-1][0:ATT_D-1];
    logic signed [15:0] c_mem [0:ATT_T-1][0:ATT_D-1];
    logic signed [15:0] o_mem [0:ATT_T-1][0:ATT_D-1];
    logic signed [15:0] p_mem [0:ATT_T-1][0:ATT_T-1];

    // row scratch (one score row / one exp row / one context accumulator row)
    logic signed [63:0] srow [0:ATT_T-1];
    logic signed [63:0] erow [0:ATT_T-1];
    logic signed [63:0] cacc [0:ATT_D-1];

    // ---------------------------------------------------------------- FSM state
    localparam int P_IDLE     = 0,
                   P_PJ_INIT  = 1,
                   P_PJ_MAC   = 2,
                   P_PJ_WR    = 3,
                   P_SC_INIT  = 4,
                   P_SC_MAC   = 5,
                   P_SC_WR    = 6,
                   P_SMAX_INIT= 7,
                   P_SMAX_L   = 8,
                   P_SEXP     = 9,
                   P_SREC     = 10,
                   P_SW_INIT  = 11,
                   P_SW_J     = 12,
                   P_SW_D     = 13,
                   P_CTX_INIT = 14,
                   P_CTX_WR   = 15,
                   P_ROW_NEXT = 16,
                   P_OUT_INIT = 17,
                   P_OUT_MAC  = 18,
                   P_OUT_WR   = 19,
                   P_DONE     = 20,
                   P_SEXP_2   = 21;

    logic [4:0]  phase;
    logic [1:0]  pk;            // projection kind 0=Q 1=K 2=V
    logic [6:0]  pi, po, pd;    // projection row / out-col / reduce
    logic [6:0]  i, j;          // row-phase query row / key col
    logic signed [63:0] acc;
    logic signed [63:0] mreg;
    logic signed [63:0] sumE;
    logic signed [63:0] Rreg;
    logic signed [63:0] Pcur;

    // ---------------------------------------------------------------- helpers
    function automatic signed [63:0] rq(input signed [63:0] v, input int s);
        if (s == 0) rq = v;
        else        rq = (v + (64'sd1 <<< (s-1))) >>> s;
    endfunction

    function automatic signed [15:0] clip16(input signed [63:0] v);
        if (v > 64'sd32767)       clip16 = 16'sd32767;
        else if (v < -64'sd32768) clip16 = -16'sd32768;
        else                      clip16 = v[15:0];
    endfunction

    logic signed [15:0] wval;
    always_comb begin
        unique case (pk)
            2'd0:    wval = wq[pd][po];
            2'd1:    wval = wk[pd][po];
            default: wval = wv[pd][po];
        endcase
    end

    logic signed [63:0] sh;
    logic signed [63:0] ka, frac;

    // ---------------------------------------------------------------- write port
    always_ff @(posedge clk) begin
        if (ld_we) begin
            unique case (ld_sel)
                3'd0: x_mem[ld_row][ld_col] <= ld_data[15:0];
                3'd1: wq[ld_row][ld_col]    <= ld_data[15:0];
                3'd2: wk[ld_row][ld_col]    <= ld_data[15:0];
                3'd3: wv[ld_row][ld_col]    <= ld_data[15:0];
                3'd4: wo[ld_row][ld_col]    <= ld_data[15:0];
                3'd5: bias_m[ld_col]        <= ld_data;
                3'd6: exp_lut[ld_row]       <= ld_data[15:0];
                default: ;
            endcase
        end
    end

    // ---------------------------------------------------------------- read port
    always_comb begin
        rd_data = 32'sd0;
        unique case (rd_sel)
            3'd0: rd_data = {{16{x_mem[rd_row][rd_col][15]}}, x_mem[rd_row][rd_col]};
            3'd1: rd_data = {{16{q_mem[rd_row][rd_col][15]}}, q_mem[rd_row][rd_col]};
            3'd2: rd_data = {{16{k_mem[rd_row][rd_col][15]}}, k_mem[rd_row][rd_col]};
            3'd3: rd_data = {{16{v_mem[rd_row][rd_col][15]}}, v_mem[rd_row][rd_col]};
            3'd4: rd_data = {{16{p_mem[rd_row][rd_col][15]}}, p_mem[rd_row][rd_col]};
            3'd5: rd_data = {{16{c_mem[rd_row][rd_col][15]}}, c_mem[rd_row][rd_col]};
            3'd6: rd_data = {{16{o_mem[rd_row][rd_col][15]}}, o_mem[rd_row][rd_col]};
            3'd7: rd_data = bias_m[rd_row];
            default: ;
        endcase
    end

    // ---------------------------------------------------------------- main FSM
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            phase <= P_IDLE;
            busy  <= 1'b0;
            done  <= 1'b0;
            pk <= 2'd0; pi <= 7'd0; po <= 7'd0; pd <= 7'd0;
            i <= 7'd0; j <= 7'd0;
            acc <= 64'sd0; mreg <= 64'sd0; sumE <= 64'sd0; Rreg <= 64'sd0; Pcur <= 64'sd0;
        end else begin
            unique case (phase)
                P_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy  <= 1'b1;
                        pk    <= 2'd0;
                        phase <= P_PJ_INIT;
                    end
                end

                // ---- Q/K/V projections: one 64-term dot per (row, col) ----------
                P_PJ_INIT: begin
                    pi <= 7'd0; po <= 7'd0; pd <= 7'd0; acc <= 64'sd0;
                    phase <= P_PJ_MAC;
                end
                P_PJ_MAC: begin
                    acc <= acc + x_mem[pi][pd] * wval;
                    if (pd == ATT_D-1) phase <= P_PJ_WR;
                    else               pd <= pd + 7'd1;
                end
                P_PJ_WR: begin
                    unique case (pk)
                        2'd0: q_mem[pi][po] <= clip16(rq(acc, RQ));
                        2'd1: k_mem[pi][po] <= clip16(rq(acc, RK));
                        default: v_mem[pi][po] <= clip16(rq(acc, RV));
                    endcase
                    acc <= 64'sd0;
                    pd  <= 7'd0;
                    if (po == ATT_D-1) begin
                        po <= 7'd0;
                        if (pi == ATT_T-1) begin
                            pi <= 7'd0;
                            if (pk == 2'd2) begin
                                i     <= 7'd0;
                                phase <= P_SC_INIT;
                            end else begin
                                pk    <= pk + 2'd1;
                                phase <= P_PJ_MAC;
                            end
                        end else begin
                            pi    <= pi + 7'd1;
                            phase <= P_PJ_MAC;
                        end
                    end else begin
                        po    <= po + 7'd1;
                        phase <= P_PJ_MAC;
                    end
                end

                // ---- scores for row i: srow[j] = Q[i] . K[j] -------------------
                P_SC_INIT: begin
                    j <= 7'd0; pd <= 7'd0; acc <= 64'sd0;
                    phase <= P_SC_MAC;
                end
                P_SC_MAC: begin
                    acc <= acc + q_mem[i][pd] * k_mem[j][pd];
                    if (pd == ATT_D-1) phase <= P_SC_WR;
                    else               pd <= pd + 7'd1;
                end
                P_SC_WR: begin
                    srow[j] <= acc;
                    acc <= 64'sd0;
                    pd  <= 7'd0;
                    if (j == ATT_T-1) begin
                        phase <= P_SMAX_INIT;
                    end else begin
                        j     <= j + 7'd1;
                        phase <= P_SC_MAC;
                    end
                end

                // ---- row max ---------------------------------------------------
                P_SMAX_INIT: begin
                    mreg <= srow[0];
                    j    <= 7'd1;
                    phase <= P_SMAX_L;
                end
                P_SMAX_L: begin
                    if (srow[j] > mreg) mreg <= srow[j];
                    if (j == ATT_T-1) begin
                        j     <= 7'd0;
                        sumE  <= 64'sd0;
                        phase <= P_SEXP;
                    end else begin
                        j <= j + 7'd1;
                    end
                end

                // ---- exp2 approximation: E = 2^((S-m)*kc) in Q(M_LUT) ----------
                P_SEXP: begin
                    ka   <= rq((srow[j] - mreg) * KC, SH_KC) >>> F_BITS;
                    frac <= rq((srow[j] - mreg) * KC, SH_KC)
                            - ((rq((srow[j] - mreg) * KC, SH_KC) >>> F_BITS) <<< F_BITS);
                    sh   <= -(rq((srow[j] - mreg) * KC, SH_KC) >>> F_BITS);
                    phase <= P_SEXP_2;
                end
                P_SEXP_2: begin
                    erow[j] <= (sh >= 64'sd32) ? 64'sd0
                              : ({{48{1'b0}}, exp_lut[frac[F_BITS-1:0]]} >>> sh);
                    sumE <= sumE + ((sh >= 64'sd32) ? 64'sd0
                              : ({{48{1'b0}}, exp_lut[frac[F_BITS-1:0]]} >>> sh));
                    if (j == ATT_T-1) phase <= P_SREC;
                    else begin
                        j     <= j + 7'd1;
                        phase <= P_SEXP;
                    end
                end

                // ---- per-row reciprocal ---------------------------------------
                P_SREC: begin
                    Rreg  <= ((64'sd1 <<< (15+G_SH)) + (sumE >>> 1)) / sumE;
                    j     <= 7'd0;
                    phase <= P_SW_INIT;
                end

                // ---- P = E*R (Q15) and fused context C[i] += P*V[j] ------------
                P_SW_INIT: begin
                    for (int t = 0; t < ATT_D; t++) cacc[t] <= 64'sd0;
                    j     <= 7'd0;
                    phase <= P_SW_J;
                end
                P_SW_J: begin
                    if (((erow[j] * Rreg) >>> G_SH) > 64'sd32767) Pcur <= 64'sd32767;
                    else Pcur <= (erow[j] * Rreg) >>> G_SH;
                    p_mem[i][j] <= (((erow[j] * Rreg) >>> G_SH) > 64'sd32767)
                                   ? 16'sd32767 : (erow[j] * Rreg) >>> G_SH;
                    pd    <= 7'd0;
                    phase <= P_SW_D;
                end
                P_SW_D: begin
                    cacc[pd] <= cacc[pd] + Pcur * v_mem[j][pd];
                    if (pd == ATT_D-1) begin
                        if (j == ATT_T-1) phase <= P_CTX_INIT;
                        else begin
                            j     <= j + 7'd1;
                            phase <= P_SW_J;
                        end
                    end else begin
                        pd <= pd + 7'd1;
                    end
                end

                // ---- context re-quantize --------------------------------------
                P_CTX_INIT: begin
                    pd <= 7'd0;
                    phase <= P_CTX_WR;
                end
                P_CTX_WR: begin
                    c_mem[i][pd] <= clip16(rq(cacc[pd], RC));
                    if (pd == ATT_D-1) phase <= P_ROW_NEXT;
                    else               pd <= pd + 7'd1;
                end

                P_ROW_NEXT: begin
                    if (i == ATT_T-1) begin
                        phase <= P_OUT_INIT;
                    end else begin
                        i     <= i + 7'd1;
                        j     <= 7'd0;
                        pd    <= 7'd0;
                        acc   <= 64'sd0;
                        phase <= P_SC_INIT;
                    end
                end

                // ---- output projection O = C @ Wo + bias -----------------------
                P_OUT_INIT: begin
                    i <= 7'd0; po <= 7'd0; pd <= 7'd0;
                    acc   <= {{32{bias_m[0][31]}}, bias_m[0]};
                    phase <= P_OUT_MAC;
                end
                P_OUT_MAC: begin
                    acc <= acc + c_mem[i][pd] * wo[pd][po];
                    if (pd == ATT_D-1) phase <= P_OUT_WR;
                    else               pd <= pd + 7'd1;
                end
                P_OUT_WR: begin
                    o_mem[i][po] <= clip16(rq(acc, RO));
                    pd <= 7'd0;
                    if (po == ATT_D-1) begin
                        po <= 7'd0;
                        if (i == ATT_T-1) begin
                            phase <= P_DONE;
                        end else begin
                            i     <= i + 7'd1;
                            acc   <= {{32{bias_m[0][31]}}, bias_m[0]};
                            phase <= P_OUT_MAC;
                        end
                    end else begin
                        po    <= po + 7'd1;
                        acc   <= {{32{bias_m[po+7'd1][31]}}, bias_m[po+7'd1]};
                        phase <= P_OUT_MAC;
                    end
                end

                P_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    if (start) begin
                        done  <= 1'b0;
                        busy  <= 1'b1;
                        pk    <= 2'd0;
                        phase <= P_PJ_INIT;
                    end
                end

                default: phase <= P_IDLE;
            endcase
        end
    end

endmodule
