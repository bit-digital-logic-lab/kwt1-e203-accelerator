// ONE shared ROWS-lane INT16 multiplier array feeding TWO independent
// ROWS x COLS INT32 accumulator (CACC) pages.
//
// This is the area-efficient version of the streaming_mac dual-CACC schedule:
// the old experiment instantiated kwt_row_broadcast_tile TWICE (2 x 16
// multipliers) purely to get two accumulator pages. Here the multiplier feed
// is instantiated ONCE and only the accumulators are duplicated, so the
// elaborated multiplier count stays at ROWS (16 for the 16-row config) instead
// of 2*ROWS (32).
//
// Clean split:
//   stage 1 "multiplier feed" : register the ROWS products + beat metadata
//                               (page, column, row-mask).
//   stage 2 "accumulator"     : add the registered products into the selected
//                               page at the next clock edge.
// Bias is added ONLY at dump/export; it never reaches the MAC input.
//
// -------------------------------------------------------------------------
// Contract (see README for the full table)
// -------------------------------------------------------------------------
// Page ownership
//   * FEED ownership: a page is fed in the cycle `in_valid && in_ready` and
//     remains feed-touched while its beat is in the product register
//     (`prod_valid_q`) and while it is accumulated at the following edge.
//   * EXPORT ownership: a page owns the export port in the cycle its accepted
//     dump request is registered (`dump_valid_q`), i.e. while its data is
//     driven on dump_data.
//   * A page may be EXPORT-owned and the other page FEED-owned at the same
//     time; the two pages never share a storage port.
// valid / ready
//   * Feed: `in_ready` is deasserted while the target page is EXPORT-owned or
//     while its `clear` bit is high. A beat is accepted iff in_valid && in_ready.
//   * Dump: `dump_ready` is deasserted while the requested page is being fed
//     this cycle, is in the product register, or has its clear bit high. The
//     feed has priority, so there is no combinational loop and dump_ready
//     never depends on itself.
// MAC latency
//   * Beat accepted at edge N -> products visible in cycle N+1
//     (`prod_q`, `prod_valid_q`) -> accumulated at edge N+1 -> CACC contents
//     visible in cycle N+1. Feed-to-CACC latency = 1 cycle (2 clock edges).
//   * A dump request accepted in cycle M returns registered data in cycle
//     M+1 (dump_valid / dump_data), so dump request-to-valid = 1 cycle.
// Dump address / data
//   * Request fields: dump_page / dump_row / dump_col / dump_bias.
//   * `dump_addr = {dump_valid_page, dump_valid_row, dump_valid_col}` is the
//     address of the datum currently driven on `dump_data`.
//   * Invalid (never written / cleared) cells export bias only.
// Busy conflicts / backpressure
//   * Dump is a 1 request/cycle streaming port; a request that is not ready is
//     a pure backpressure event and must be retried. Request gaps are allowed
//     and simply produce idle dump_valid cycles.
//   * Feeding the page that currently owns export is a conflict: in_ready=0.
//   * Dumping a page that is being fed is a conflict: dump_ready=0.
// Clear / reset
//   * `clear[p]` synchronously invalidates page p's valid bits; a product for
//     page p arriving in the same edge is dropped and a dump delivery for page
//     p is cancelled (`dump_valid` is gated by clear).
//   * `rst_n` is asynchronous active-low: both pages, both pipeline stages and
//     the export register are cleared.
module kwt_single_mac_dual_cacc #(
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int DATA_W = 16,
    parameter int ACC_W = 32,
    parameter int PAGES = 2,
    parameter int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter int COL_W = (COLS <= 2) ? 1 : $clog2(COLS),
    parameter int PAGE_W = (PAGES <= 2) ? 1 : $clog2(PAGES)
) (
    input  logic clk,
    input  logic rst_n,
    // ---------------- shared multiplier feed ----------------
    input  logic in_valid,
    output logic in_ready,
    input  logic [PAGE_W-1:0] in_page,
    input  logic [ROWS-1:0] row_mask,
    input  logic [ROWS*DATA_W-1:0] a_rows,
    input  logic [DATA_W-1:0] b_word,
    input  logic [COL_W-1:0] in_col,
    // ---------------- synchronous per-page clear ----------------
    input  logic [PAGES-1:0] clear,
    // ---------------- export / dump port (bias added here) ----------------
    input  logic dump_req,
    output logic dump_ready,
    input  logic [PAGE_W-1:0] dump_page,
    input  logic [ROW_W-1:0] dump_row,
    input  logic [COL_W-1:0] dump_col,
    input  logic [ACC_W-1:0] dump_bias,
    output logic dump_valid,
    output logic [PAGE_W-1:0] dump_valid_page,
    output logic [ROW_W-1:0] dump_valid_row,
    output logic [COL_W-1:0] dump_valid_col,
    output logic [PAGE_W+ROW_W+COL_W-1:0] dump_addr,
    output logic [ACC_W-1:0] dump_data
);
    localparam int CELLS = ROWS * COLS;
    localparam int DEPTH = PAGES * CELLS;

    // stage 1: multiplier feed registers
    logic prod_valid_q;
    logic [PAGE_W-1:0] prod_page_q;
    logic [COL_W-1:0] prod_col_q;
    logic [ROWS-1:0] prod_mask_q;
    logic signed [2*DATA_W-1:0] prod_q [0:ROWS-1];

    // stage 2: accumulator storage (one valid bit + one INT32 per element/page)
    logic signed [ACC_W-1:0] psum_mem [0:DEPTH-1];
    logic [PAGES-1:0][CELLS-1:0] vbits;

    // export register
    logic dump_valid_q;
    logic [PAGE_W-1:0] dump_valid_page_q;
    logic [ROW_W-1:0] dump_valid_row_q;
    logic [COL_W-1:0] dump_valid_col_q;
    logic [ACC_W-1:0] dump_data_q;

    // ---- page ownership / valid-ready -------------------------------------
    // Feed has priority over dump in a same-cycle collision, so in_ready does
    // not depend on dump_ready (no combinational loop).
    assign in_ready = rst_n && !clear[in_page] &&
                      !(dump_valid_q && (dump_valid_page_q == in_page));

    assign dump_ready = rst_n && !clear[dump_page] &&
                        !(in_valid && in_ready && (in_page == dump_page)) &&
                        !(prod_valid_q && (prod_page_q == dump_page));

    assign dump_valid = dump_valid_q && !clear[dump_valid_page_q];
    assign dump_addr = {dump_valid_page_q, dump_valid_row_q, dump_valid_col_q};
    assign dump_valid_page = dump_valid_page_q;
    assign dump_valid_row  = dump_valid_row_q;
    assign dump_valid_col  = dump_valid_col_q;
    assign dump_data = dump_data_q;

    // ---- stage 1: shared multiplier array (ROWS multipliers, ONE copy) ----
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_valid_q <= 1'b0;
            prod_page_q <= '0;
            prod_col_q <= '0;
            prod_mask_q <= '0;
            for (int r = 0; r < ROWS; r++) prod_q[r] <= '0;
        end else begin
            prod_valid_q <= in_valid && in_ready;
            if (in_valid && in_ready) begin
                prod_page_q <= in_page;
                prod_col_q  <= in_col;
                prod_mask_q <= row_mask;
                for (int r = 0; r < ROWS; r++)
                    prod_q[r] <= $signed(a_rows[r*DATA_W +: DATA_W]) *
                                 $signed(b_word);
            end
        end
    end

    // ---- stage 2: accumulator write + synchronous per-page clear -----------
    // Single driver block for the psum/valid storage (avoids MULTIDRIVEN).
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vbits <= '0;
        end else begin
            for (int p = 0; p < PAGES; p++)
                if (clear[p]) vbits[p] <= '0;
            for (int r = 0; r < ROWS; r++) begin
                if (prod_valid_q && prod_mask_q[r] && !clear[prod_page_q]) begin
                    if (vbits[prod_page_q][r*COLS + int'(prod_col_q)])
                        psum_mem[(int'(prod_page_q)*ROWS + r)*COLS + int'(prod_col_q)]
                            <= psum_mem[(int'(prod_page_q)*ROWS + r)*COLS + int'(prod_col_q)]
                               + ACC_W'(prod_q[r]);
                    else
                        psum_mem[(int'(prod_page_q)*ROWS + r)*COLS + int'(prod_col_q)]
                            <= ACC_W'(prod_q[r]);
                    vbits[prod_page_q][r*COLS + int'(prod_col_q)] <= 1'b1;
                end
            end
        end
    end

    // ---- export register (reads psum/vbits; never writes them) ------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dump_valid_q <= 1'b0;
            dump_valid_page_q <= '0;
            dump_valid_row_q <= '0;
            dump_valid_col_q <= '0;
            dump_data_q <= '0;
        end else begin
            dump_valid_q <= dump_req && dump_ready;
            if (dump_req && dump_ready) begin
                dump_valid_page_q <= dump_page;
                dump_valid_row_q  <= dump_row;
                dump_valid_col_q  <= dump_col;
                // Bias is injected here, at export, never at the MAC input.
                if (vbits[dump_page][int'(dump_row)*COLS + int'(dump_col)])
                    dump_data_q <= psum_mem[(int'(dump_page)*ROWS + int'(dump_row))*COLS
                                            + int'(dump_col)] + dump_bias;
                else
                    dump_data_q <= dump_bias;
            end
        end
    end
endmodule
