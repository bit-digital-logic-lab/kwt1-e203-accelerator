// One 16-bit word/slot per cycle shared-bus arbiter.
//
// This is deliberately small and synthesizable. It is NOT a memory controller:
// it only resolves which local DMA/engine may drive the single external 16-bit
// data bus in a cycle. The local SRAM banks behind each requester may still
// perform their own read/write ports concurrently; that is a separate resource.
//
// Priority is fixed by request index (index 0 highest). The bias-overlap
// controller uses:
//   0 OUT   - result write-back (must not overflow the output stage)
//   1 BIAS  - bias column-group fill (has a per-tile export deadline)
//   2 A     - resident-A fill / row-tile streaming
//   3 W     - next-weight-page prefetch
//
// Contract: a requester holds req[i] and its data/address stable until it sees
// gnt[i]. Exactly one gnt bit can be high in a cycle while enable=1. OUT and
// BIAS are finite bursts; A is finite; W is finite per page, so the fixed
// priority cannot starve a requester indefinitely. enable=0 inserts a global
// bus-contention/idle cycle without losing any held request.
module kwt_shared_bus_arbiter #(
    parameter int N = 4
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         enable,
    input  logic [N-1:0] req,
    output logic [N-1:0] gnt
);
    always_comb begin
        gnt = '0;
        if (rst_n && enable) begin
            for (int i = 0; i < N; i++) begin
                if (req[i] && (gnt == '0))
                    gnt[i] = 1'b1;
            end
        end
    end

`ifdef SIMULATION_ASSERT
    // A one-hot check that is also useful in the unit microtest.
    always_ff @(posedge clk) begin
        if (rst_n && enable && ($countones(gnt) > 1))
            $fatal(1, "arbiter granted more than one requester");
    end
`endif
endmodule
