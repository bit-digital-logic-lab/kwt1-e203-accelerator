// Top assembly: single-layer PostNorm encoder datapath + 12-layer scheduling
// controller.  This is the hardware boundary the E203 software driver talks to
// (see graduate/encoder/SOFTWARE_BOUNDARY.md).  There is no arithmetic in this
// file: it only wires the controller's transaction stream to the layer's load
// port and routes the layer output stream.
module kwt_encoder_top #(
    parameter int T   = 99,
    parameter int D   = 64,
    parameter int HID = 256,
    parameter int LAYERS = 12,
    parameter int M_A = 5981,
    parameter int Q_A = 16
) (
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic                 start,
    output logic                 busy,
    output logic                 done,

    // parameter/transport source (E203-side region)
    input  logic signed [31:0]   p_rdata,

    // classification-head boundary (software stage follows)
    output logic                 head_valid,
    input  logic                 head_ready,
    output logic signed [D*16-1:0] head_token,
    output logic [3:0]           layer_id,

    // layer output stream (also used for the 12-layer smoke observation)
    output logic                 y_valid,
    output logic signed [D*16-1:0] y_data,

    // controller transaction port (what the E203 transport model observes)
    output logic [3:0]           o_ld_sel,
    output logic [13:0]          o_ld_addr,
    output logic                 o_ld_we,
    output logic                 o_enc_start,
    output logic                 o_y_ready,

    // per-stage debug snapshots (level-by-level comparison)
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

    logic [3:0]         u_ld_sel;
    logic [13:0]        u_ld_addr;
    logic signed [31:0] u_ld_data;
    logic               u_ld_we;
    logic               enc_start, enc_busy, enc_done;
    logic               ctrl_y_ready;

    kwt_encoder_layer #(.T(T), .D(D), .HID(HID), .M_A(M_A), .Q_A(Q_A)) u_layer (
        .clk, .rst_n,
        .ld_sel(u_ld_sel), .ld_addr(u_ld_addr), .ld_data(u_ld_data), .ld_we(u_ld_we),
        .start(enc_start), .busy(enc_busy), .done(enc_done),
        .y_valid(y_valid), .y_ready(ctrl_y_ready), .y_data(y_data),
        .dbg_a_valid, .dbg_a, .dbg_r1_valid, .dbg_r1,
        .dbg_g_valid, .dbg_g, .dbg_f_valid, .dbg_f, .dbg_r2_valid, .dbg_r2,
        .o1_sat_count, .y_sat_count);

    kwt_encoder_layers_ctrl #(.T(T), .D(D), .HID(HID), .LAYERS(LAYERS)) u_ctrl (
        .clk, .rst_n,
        .start, .busy, .done,
        .u_ld_sel, .u_ld_addr, .u_ld_data, .u_ld_we,
        .enc_start, .enc_busy, .enc_done,
        .y_valid, .y_ready(ctrl_y_ready), .y_data,
        .p_rdata,
        .head_valid, .head_ready, .head_token, .layer_id);

    assign o_ld_sel  = u_ld_sel;
    assign o_ld_addr = u_ld_addr;
    assign o_ld_we   = u_ld_we;
    assign o_enc_start = enc_start;
    assign o_y_ready   = ctrl_y_ready;

endmodule
