// Dedicated bias bank for the bias-at-CACC-export schedule.
//
// Storage: DEPTH INT32 values (256 for one FFN1 layer = 1 KiB).
// External fill: one 16-bit half-word per bus slot, so one INT32 bias value
// takes two slots. A column group is COLS=16 consecutive INT32 values, i.e.
// 32 shared-bus slots. group_valid[g] rises only after all 32 half-writes of
// that group have landed, so a partially loaded group can never be used by an
// export. The bank has a separate read port for the output stage; the fill
// port and the read port are allowed to operate in the same cycle because
// they are different local SRAM ports. They must never target the same group
// in a way that would expose a half-written value: the controller only starts
// an export after group_valid[g] is high.
//
// clear (not rst_n) is the task-level invalidation. Data bits are not reset
// (SRAM-inferable); validity is. This prevents stale bias after a task reset.
module kwt_bias_bank #(
    parameter int DEPTH  = 256,
    parameter int COLS   = 16,
    parameter int GROUPS = DEPTH / COLS,
    parameter int GW     = (GROUPS <= 2) ? 1 : $clog2(GROUPS),
    parameter int IW     = (COLS <= 2) ? 1 : $clog2(COLS),
    parameter int AW     = (DEPTH <= 2) ? 1 : $clog2(DEPTH)
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                clear,
    // 16-bit fill port (shared bus side)
    input  logic                wr_en,
    input  logic [GW-1:0]       wr_group,
    input  logic [IW-1:0]       wr_idx,
    input  logic                wr_half,
    input  logic [15:0]         wr_data,
    // 32-bit read port (CACC export side)
    input  logic [AW-1:0]       rd_col,
    output logic [31:0]         rd_data,
    output logic [GROUPS-1:0]   group_valid
);
    logic [31:0] mem [0:DEPTH-1];
    logic [GROUPS-1:0][COLS-1:0][1:0] half_done;

    always_ff @(posedge clk) begin
        if (wr_en) begin
            if (wr_half)
                mem[int'(wr_group) * COLS + int'(wr_idx)][31:16] <= wr_data;
            else
                mem[int'(wr_group) * COLS + int'(wr_idx)][15:0] <= wr_data;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            half_done <= '0;
        else if (clear)
            half_done <= '0;
        else if (wr_en)
            half_done[wr_group][wr_idx][wr_half] <= 1'b1;
    end

    assign rd_data = mem[rd_col];

    for (genvar g = 0; g < GROUPS; g++) begin : g_valid
        assign group_valid[g] = &half_done[g];
    end
endmodule
