// Focused RTL microtest for the two new primitives:
//   1. kwt_bias_bank: 16-bit half-word fill, group_valid only after 32 slots,
//      correct INT32 read, clear/reset invalidation, no partial-group use.
//   2. kwt_shared_bus_arbiter: exactly one grant per enabled cycle, fixed
//      priority, held requests survive back-pressure, reset disables grants.
//
// It also runs a delayed-bias producer/consumer: the producer is held off for
// a while; the consumer must not start before group_valid and must then read
// the exact INT32 values. This is the deadline behaviour used by the task TB.
module tb_kwt_bias_arbiter_unit;
    localparam int DEPTH = 32, COLS = 16, GROUPS = 2;
    localparam int AW = $clog2(DEPTH), GW = 1, IW = 4;

    logic clk = 0, rst_n = 0, clear = 0;
    logic wr_en = 0;
    logic [GW-1:0] wr_group = '0;
    logic [IW-1:0] wr_idx = '0;
    logic wr_half = 0;
    logic [15:0] wr_data = '0;
    logic [AW-1:0] rd_col = '0;
    logic [31:0] rd_data;
    logic [GROUPS-1:0] group_valid;

    logic enable = 1;
    logic [3:0] req = '0, gnt;

    int errors = 0;
    int cyc = 0;
    logic [31:0] expect_mem [0:DEPTH-1];

    always #5 clk = ~clk;
    initial begin
        #200000;
        $fatal(1, "bias/arbiter unit watchdog expired");
    end

    kwt_bias_bank #(.DEPTH(DEPTH), .COLS(COLS), .GROUPS(GROUPS)) bank (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .wr_en(wr_en), .wr_group(wr_group), .wr_idx(wr_idx),
        .wr_half(wr_half), .wr_data(wr_data),
        .rd_col(rd_col), .rd_data(rd_data), .group_valid(group_valid)
    );

    kwt_shared_bus_arbiter #(.N(4)) arb (
        .clk(clk), .rst_n(rst_n), .enable(enable), .req(req), .gnt(gnt)
    );

    task automatic write_one(input int g, input int idx, input int val, input int half);
        @(negedge clk);
        wr_en = 1'b1;
        wr_group = GW'(g);
        wr_idx = IW'(idx);
        wr_half = half[0];
        wr_data = half ? 16'((val >> 16) & 16'hFFFF) : 16'(val & 16'hFFFF);
        @(negedge clk);
        wr_en = 1'b0;
    endtask

    task automatic write_group(input int g);
        for (int i = 0; i < COLS; i++) begin
            write_one(g, i, expect_mem[g*COLS+i], 0);
            write_one(g, i, expect_mem[g*COLS+i], 1);
        end
    endtask

    task automatic check_group(input int g);
        for (int i = 0; i < COLS; i++) begin
            rd_col = AW'(g*COLS+i);
            #1;
            if (rd_data !== expect_mem[g*COLS+i]) begin
                $error("bias read g=%0d i=%0d got=%h exp=%h", g, i, rd_data,
                       expect_mem[g*COLS+i]);
                errors++;
            end
        end
    endtask

    // One cycle of the delayed producer/consumer test. Returns 1 when the
    // producer has finished this group.
    task automatic producer_step(input int g, input int delay, inout int state,
                                  inout int idx, inout int half, output bit done);
        done = 1'b0;
        if (state == 0) begin
            if (cyc >= delay) state = 1;
        end else if (state == 1) begin
            wr_en = 1'b1;
            wr_group = GW'(g);
            wr_idx = IW'(idx);
            wr_half = half[0];
            wr_data = half ? 16'((expect_mem[g*COLS+idx] >> 16) & 16'hFFFF)
                           : 16'(expect_mem[g*COLS+idx] & 16'hFFFF);
            if (half) begin
                half = 0;
                if (idx == COLS - 1) begin
                    state = 2;
                    idx = 0;
                end else idx++;
            end else half = 1;
        end
        if (state == 2) begin
            // Leave the final half-write asserted; the caller lets the next
            // posedge commit it before dropping wr_en.
            done = 1'b1;
        end
    endtask

    initial begin : run
        int state, pidx, phalf, waited, started;
        bit producer_done;
        // Deterministic expected bias values, including INT32 sign/extremes.
        for (int g = 0; g < GROUPS; g++)
            for (int i = 0; i < COLS; i++)
                expect_mem[g*COLS+i] = (g * 32'h00001111 + i * 32'h00000101) ^ 32'h5A5A0000;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // ---- bias bank normal fill / validity ----
        if (group_valid !== 2'b00) begin $error("valid not clear after reset"); errors++; end
        for (int i = 0; i < COLS; i++) begin
            write_one(0, i, expect_mem[i], 0);
            if (group_valid[0] !== 1'b0) begin
                $error("group_valid set with only low halves at i=%0d", i); errors++;
            end
        end
        for (int i = 0; i < COLS; i++)
            write_one(0, i, expect_mem[i], 1);
        @(negedge clk);
        if (group_valid[0] !== 1'b1) begin $error("group0 did not become valid"); errors++; end
        check_group(0);

        // Partial group1 must not be valid, group0 must stay valid.
        for (int i = 0; i < 8; i++) begin
            write_one(1, i, expect_mem[COLS+i], 0);
            write_one(1, i, expect_mem[COLS+i], 1);
        end
        @(negedge clk);
        if (group_valid[1] !== 1'b0) begin $error("partial group1 became valid"); errors++; end
        if (group_valid[0] !== 1'b1) begin $error("group0 lost validity"); errors++; end
        for (int i = 8; i < COLS; i++) begin
            write_one(1, i, expect_mem[COLS+i], 0);
            write_one(1, i, expect_mem[COLS+i], 1);
        end
        @(negedge clk);
        if (group_valid[1] !== 1'b1) begin $error("group1 did not become valid"); errors++; end
        check_group(1);

        // ---- clear invalidates only validity ----
        @(negedge clk); clear = 1'b1;
        @(negedge clk); clear = 1'b0;
        @(negedge clk);
        if (group_valid !== 2'b00) begin $error("clear did not invalidate groups"); errors++; end

        // ---- reset during a partial fill ----
        write_one(0, 0, expect_mem[0], 0);
        write_one(0, 0, expect_mem[0], 1);
        write_one(0, 1, expect_mem[1], 0);
        @(negedge clk); rst_n = 1'b0;
        repeat (2) @(negedge clk);
        if (group_valid !== 2'b00) begin $error("reset did not clear validity"); errors++; end
        rst_n = 1'b1;
        @(negedge clk);
        write_group(0);
        @(negedge clk);
        if (group_valid[0] !== 1'b1) begin $error("group0 invalid after reset refill"); errors++; end
        check_group(0);

        // ---- delayed producer / consumer must wait for valid ----
        @(negedge clk); clear = 1'b1;
        @(negedge clk); clear = 1'b0;
        state = 0; pidx = 0; phalf = 0; waited = 0; started = 0;
        for (int t = 0; t < 200; t++) begin
            @(negedge clk);
            cyc = t;
            wr_en = 1'b0;
            if (!group_valid[1] && !started) waited++;
            producer_step(1, 40, state, pidx, phalf, producer_done);
            if (producer_done) begin
                // producer done this cycle; the next posedge commits the last half
                @(negedge clk);
                wr_en = 1'b0;
                repeat (2) @(negedge clk);
                if (group_valid[1] !== 1'b1) begin
                    $error("delayed producer did not make group1 valid (valid=%b)", group_valid); errors++;
                end
                started = 1;
                break;
            end
        end
        if (!started) begin $error("delayed producer did not finish"); errors++; end
        if (waited < 30) begin $error("consumer did not observe the bias delay"); errors++; end
        check_group(1);

        // ---- arbiter: one-hot, fixed priority, held request, reset ----
        @(negedge clk);
        req = 4'b1111; enable = 1;
        repeat (3) begin
            @(negedge clk);
            if (gnt !== 4'b0001) begin $error("priority OUT failed gnt=%b", gnt); errors++; end
            if ($countones(gnt) > 1) begin $error("multi-grant"); errors++; end
        end
        req = 4'b1110;
        @(negedge clk);
        if (gnt !== 4'b0010) begin $error("priority BIAS failed gnt=%b", gnt); errors++; end
        req = 4'b1100;
        @(negedge clk);
        if (gnt !== 4'b0100) begin $error("priority A failed gnt=%b", gnt); errors++; end
        req = 4'b1000;
        @(negedge clk);
        if (gnt !== 4'b1000) begin $error("priority W failed gnt=%b", gnt); errors++; end
        req = 4'b0000;
        @(negedge clk);
        if (gnt !== 4'b0000) begin $error("grant without request"); errors++; end
        req = 4'b1111; enable = 0;
        @(negedge clk);
        if (gnt !== 4'b0000) begin $error("grant while disabled"); errors++; end
        enable = 1;
        @(negedge clk); rst_n = 1'b0;
        @(negedge clk);
        if (gnt !== 4'b0000) begin $error("grant during reset"); errors++; end
        rst_n = 1'b1;

        // Held A request: exactly one grant per enabled cycle, no duplicates.
        req = 4'b0100;
        begin
            int grants;
            grants = 0;
            for (int t = 0; t < 12; t++) begin
                @(negedge clk);
                if (gnt[2]) grants++;
                if ($countones(gnt) > 1) begin $error("multi-grant under hold"); errors++; end
            end
            if (grants != 12) begin $error("held request grants=%0d expected 12", grants); errors++; end
        end

        if (errors != 0) $fatal(1, "bias/arbiter unit FAILED with %0d errors", errors);
        $display("PASS bias/arbiter unit: bank validity/reset, delayed bias, one-hot arbiter, back-pressure");
        $finish;
    end
endmodule
