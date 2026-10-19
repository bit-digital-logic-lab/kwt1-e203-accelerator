// Second feed-forward projection of the KWT-1 encoder: 256 -> 64, INT32 accumulate.
//
//   f[k] = sat_int32( sum_j g_int[j] * w2[k][j] + b2_int[k] )
//
// g_int is the GELU output in scale s_g, w2 is symmetric INT16 with a
// per-output-channel scale s_w2[k] and s_h2[k] = s_g * s_w2[k].  w2_int and
// b2_int are loaded through a simple synchronous write port (behavioural SRAM,
// the real design is filled by the per-layer parameter DMA).
//
// The 64 accumulators are updated in parallel while j sweeps 0..255, one
// hidden index per cycle, mirroring the row-broadcast style of the existing
// FFN1 core without modifying it.
module kwt_ffn2 #(
    parameter int HID   = 256,
    parameter int OUT   = 64,
    parameter int IN_W  = 16,
    parameter int W_W   = 16,
    parameter int ACC_W = 48,
    parameter int OUT_W = 32
) (
    input  logic clk,
    input  logic rst_n,
    // weight / bias load port
    input  logic w_we,
    input  logic [$clog2(HID*OUT)-1:0] w_addr,
    input  logic signed [W_W-1:0] w_wdata,
    input  logic b_we,
    input  logic [$clog2(OUT)-1:0] b_addr,
    input  logic signed [OUT_W-1:0] b_wdata,
    // execution
    input  logic start,
    input  logic signed [HID*IN_W-1:0] g_data,
    output logic busy,
    output logic done,
    output logic signed [OUT*OUT_W-1:0] f_data
);
    logic signed [W_W-1:0] w2 [0:HID*OUT-1];
    logic signed [OUT_W-1:0] b2 [0:OUT-1];
    logic signed [IN_W-1:0] g_buf [0:HID-1];
    logic signed [ACC_W-1:0] acc [0:OUT-1];
    logic signed [ACC_W-1:0] nxt [0:OUT-1];
    logic signed [ACC_W-1:0] fin [0:OUT-1];
    logic [8:0] j;
    logic running;

    always_ff @(posedge clk) begin
        if (w_we) w2[w_addr] <= w_wdata;
        if (b_we) b2[b_addr] <= b_wdata;
    end

    always_comb begin
        for (int k = 0; k < OUT; k++) begin
            nxt[k] = acc[k] + ACC_W'($signed(g_buf[j]) * $signed(w2[k*HID + j]));
            fin[k] = nxt[k] + ACC_W'($signed(b2[k]));
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done    <= 1'b0;
            j       <= 9'd0;
            for (int k = 0; k < OUT; k++) acc[k] <= '0;
        end else begin
            done <= 1'b0;
            if (!running) begin
                if (start) begin
                    for (int i = 0; i < HID; i++)
                        g_buf[i] <= g_data[i*IN_W +: IN_W];
                    for (int k = 0; k < OUT; k++)
                        acc[k] <= '0;
                    j       <= 9'd0;
                    running <= 1'b1;
                end
            end else if (j == HID[8:0] - 9'd1) begin
                running <= 1'b0;
                done    <= 1'b1;
                for (int k = 0; k < OUT; k++) begin
                    if (fin[k] > 48'sd2147483647)
                        f_data[k*OUT_W +: OUT_W] <= 32'sh7fffffff;
                    else if (fin[k] < -48'sd2147483648)
                        f_data[k*OUT_W +: OUT_W] <= 32'sh80000000;
                    else
                        f_data[k*OUT_W +: OUT_W] <= fin[k][OUT_W-1:0];
                end
            end else begin
                for (int k = 0; k < OUT; k++)
                    acc[k] <= nxt[k];
                j <= j + 9'd1;
            end
        end
    end

    assign busy = running;
endmodule
