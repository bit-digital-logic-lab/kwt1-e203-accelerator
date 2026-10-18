// NICE-like protocol boundary verification for the provisional
// kwt_nice_system_model integration model.
//
// Scope: what THIS RTL actually exposes (see kwt_nice_system_model.sv and
// kwt_tile_system_model.sv). This tb does NOT claim the official E203 NICE
// protocol or the competition SoC is verified; it drives the same ports the
// on-chip substitute `e203_subsys_nice_core` drives and checks the
// valid/ready handshake, response hold, request backpressure, resets,
// illegal / out-of-range descriptors and DMA error/done reporting.
//
// Fault injection (compile-time, for the negative runs of run_nice_protocol.sh):
//   -DNICE_FAULT_RSP : force the DUT response payload wrong (DUT fault).
//   -DNICE_FAULT_BUS : inject address instability while a bus beat is stalled
//                      (exercises the bus-stability monitor itself).
`timescale 1ns/1ps
module tb_nice_protocol_boundaries;

    localparam int EXT_MEM_BYTES = 4096; // 0x1000, keeps rs1 boundary testable
    localparam logic [31:0] GOOD_INSTR = 32'h0000_600b; // custom-0, funct3=110, funct7=0
    localparam int OUT_BASE = 'h300;
    localparam int DESC_BASE = 'h500;

    logic clk = 0, rst_n = 0;
    logic nice_req_valid = 0, nice_req_ready;
    logic [31:0] nice_req_instr = GOOD_INSTR, nice_req_rs1 = DESC_BASE, nice_req_rs2 = 0;
    logic nice_rsp_valid, nice_rsp_ready = 0, nice_rsp_err;
    logic [31:0] nice_rsp_data;
    logic bus_valid, bus_write, bus_ready;
    logic [31:0] bus_addr;
    logic [127:0] bus_wdata, bus_rdata;
    logic [15:0] bus_wstrb;
    logic bus_rvalid;

    int scenarios = 0;
    int commands = 0;
    int assertions = 0;
    int bus_beats = 0;

    always #5 clk = ~clk;

    kwt_nice_system_model #(.EXT_MEM_BYTES(EXT_MEM_BYTES)) dut (.*);

    // ------------------------------------------------------------------
    // Controlled external memory with optional request backpressure.
    // ------------------------------------------------------------------
    logic [127:0] mem [0:EXT_MEM_BYTES/16-1];
    logic bp_enable = 0;
    logic bp_ready = 1;
    int bp_cnt = 0;

    assign bus_ready = bp_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bp_cnt <= 0;
            bp_ready <= 1'b1;
        end else begin
            bp_cnt <= bp_cnt + 1;
            // Stall 2 of every 5 cycles to exercise valid/payload hold.
            bp_ready <= bp_enable ? ((bp_cnt % 5) > 1) : 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bus_rvalid <= 1'b0;
            bus_rdata <= '0;
        end else begin
            bus_rvalid <= bus_valid && bus_ready && !bus_write;
            if (bus_valid && bus_ready && (bus_addr < EXT_MEM_BYTES) && (bus_addr[3:0] == 0)) begin
                if (bus_write) begin
                    for (int i = 0; i < 16; i++)
                        if (bus_wstrb[i])
                            mem[bus_addr[15:4]][i*8 +: 8] <= bus_wdata[i*8 +: 8];
                end else begin
                    bus_rdata <= mem[bus_addr[15:4]];
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Monitored bus view: identical to the real bus unless a fault is injected.
    // ------------------------------------------------------------------
    logic [31:0] obs_addr;
`ifdef NICE_FAULT_BUS
    int f_cnt = 0;
    always_ff @(posedge clk) if (rst_n) f_cnt <= f_cnt + 1;
    // Corrupt the address only while a beat is stalled, so the stability
    // monitor must observe a payload change on a non-accepted beat.
    assign obs_addr = (bus_valid && !bus_ready && f_cnt[0]) ? (bus_addr ^ 32'h10) : bus_addr;
`else
    assign obs_addr = bus_addr;
`endif

    // ------------------------------------------------------------------
    // Bus stability monitor: a presented beat must be held until accepted.
    // ------------------------------------------------------------------
    logic bs_pending;
    logic [31:0] bs_addr;
    logic [127:0] bs_wdata;
    logic [15:0] bs_wstrb;
    logic bs_write;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bs_pending <= 1'b0;
            bs_addr <= '0;
            bs_wdata <= '0;
            bs_wstrb <= '0;
            bs_write <= 1'b0;
        end else begin
            if (bs_pending) begin
                assertions++;
                if (bus_valid !== 1'b1 || obs_addr !== bs_addr || bus_write !== bs_write ||
                    bus_wdata !== bs_wdata || bus_wstrb !== bs_wstrb)
                    $fatal(1, "BUS_STABILITY_VIOLATION valid=%b addr=%h(want %h) write=%b",
                           bus_valid, obs_addr, bs_addr, bus_write);
            end
            bs_pending <= bus_valid && !bus_ready;
            if (bus_valid && !bus_ready) begin
                bs_addr <= obs_addr;
                bs_wdata <= bus_wdata;
                bs_wstrb <= bus_wstrb;
                bs_write <= bus_write;
            end
            if (bus_valid && bus_ready) bus_beats++;
        end
    end

    // ------------------------------------------------------------------
    // NICE response hold monitor: while a response is not accepted, it must
    // stay valid and keep the same payload.
    // ------------------------------------------------------------------
    logic rs_pending;
    logic [31:0] rs_data;
    logic rs_err;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rs_pending <= 1'b0;
            rs_data <= '0;
            rs_err <= 1'b0;
        end else begin
            if (rs_pending) begin
                assertions++;
                if (nice_rsp_valid !== 1'b1 || nice_rsp_data !== rs_data || nice_rsp_err !== rs_err)
                    $fatal(1, "RESPONSE_HOLD_VIOLATION valid=%b data=%h(want %h) err=%b(want %b)",
                           nice_rsp_valid, nice_rsp_data, rs_data, nice_rsp_err, rs_err);
            end
            rs_pending <= nice_rsp_valid && !nice_rsp_ready;
            rs_data <= nice_rsp_data;
            rs_err <= nice_rsp_err;
        end
    end

`ifdef NICE_FAULT_RSP
    // Injected DUT fault: the response payload is stuck wrong. A correct
    // checker must reject the very first comparison against it.
    initial force dut.nice_rsp_data = 32'hdead_beef;
`endif

    // ==================================================================
    // Stimulus helpers
    // ==================================================================
    task automatic put_h(input int ba, input logic [15:0] w);
        mem[ba>>4][(((ba%16)/2)*16) +: 16] = w;
    endtask

    task automatic put_w(input int ba, input logic [31:0] w);
        mem[ba>>4][(((ba%16)/4)*32) +: 32] = w;
    endtask

    task automatic set_desc(input int base, input logic [31:0] a, b, bias, outp);
        mem[base>>4] = {outp, bias, b, a};
    endtask

    task automatic set_desc2(input int base, input logic [6:0] klen, input logic relu,
                             input logic [7:0] rm, input logic [15:0] cm);
        // [6:0]=k_len, [7]=relu, [39:32]=row_mask, [55:40]=col_mask.
        mem[(base+16)>>4] = {64'b0, 8'b0, cm, rm, 24'b0, relu, klen};
    endtask

    task automatic init_operands();
        for (int k = 0; k < 2; k++) begin
            for (int r = 0; r < 8; r++)
                put_h(0 + (k*8+r)*2, 16'(k == 0 ? r+1 : -(r+1)));
            for (int c = 0; c < 16; c++)
                put_h('h100 + (k*16+c)*2, 16'(c+1+k));
        end
        for (int c = 0; c < 16; c++)
            put_w('h200 + c*4, 32'(c-3));
    endtask

    task automatic write_default_descriptor(input int base, input [6:0] klen);
        set_desc(base, 0, 'h100, 'h200, OUT_BASE);
        set_desc2(base, klen, 1'b1, 8'h7f, 16'h7fff);
    endtask

    task automatic check_output(input logic [7:0] rm, input logic [15:0] cm);
        for (int r = 0; r < 8; r++) begin
            for (int c = 0; c < 16; c++) begin
                int signed expected;
                logic [31:0] actual;
                int ba;
                expected = c-r-4;
                if (expected < 0 || !rm[r] || !cm[c]) expected = 0;
                ba = OUT_BASE + (r*16+c)*4;
                actual = mem[ba>>4][(((ba%16)/4)*32) +: 32];
                assertions++;
                if (actual !== 32'(expected))
                    $fatal(1, "OUTPUT_MISMATCH [%0d,%0d]=%h expected=%h", r, c, actual, 32'(expected));
            end
        end
    endtask

    // Issue one command, hold the response under CPU backpressure for `hold`
    // cycles, then accept it and check the status.
    task automatic run_cmd(input logic [31:0] instr, input logic [31:0] rs1,
                           input logic [1:0] exp_status, input int hold);
        int cyc;
        commands++;
        while (!nice_req_ready) @(negedge clk);
        @(negedge clk);
        nice_req_valid = 1;
        nice_req_instr = instr;
        nice_req_rs1 = rs1;
        @(negedge clk);
        nice_req_valid = 0;
        for (cyc = 0; !nice_rsp_valid && cyc <= 40000; cyc = cyc + 1) @(negedge clk);
        if (!nice_rsp_valid) $fatal(1, "TIMEOUT waiting for NICE response");
        for (int i = 0; i < hold; i++) begin
            assertions++;
            if (!nice_rsp_valid) $fatal(1, "RESPONSE_DROPPED during backpressure hold");
            @(negedge clk);
        end
        assertions++;
        if (nice_rsp_data !== {30'b0, exp_status} || nice_rsp_err !== (exp_status != 0))
            $fatal(1, "RSP_VALUE_MISMATCH data=%h err=%b expected_status=%0d",
                   nice_rsp_data, nice_rsp_err, exp_status);
        nice_rsp_ready = 1;
        @(negedge clk);
        nice_rsp_ready = 0;
    endtask

    // ==================================================================
    // Scenarios
    // ==================================================================
    initial begin
        init_operands();
        write_default_descriptor(DESC_BASE, 7'd2);

        repeat (3) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        // S01 reset release state.
        scenarios++;
        assertions++;
        if (nice_rsp_valid !== 1'b0) $fatal(1, "RESPONSE_ACTIVE_AFTER_RESET");
        assertions++;
        if (nice_req_ready !== 1'b1) $fatal(1, "REQUEST_NOT_READY_AFTER_RESET");

        // S02 normal transaction, response held 5 cycles, output verified.
        scenarios++;
        run_cmd(GOOD_INSTR, DESC_BASE, 2'd0, 5);
        check_output(8'h7f, 16'h7fff);

        // S03 error response (command error) held 4 cycles.
        scenarios++;
        set_desc2(DESC_BASE, 7'd0, 1'b1, 8'h7f, 16'h7fff); // k_len = 0
        run_cmd(GOOD_INSTR, DESC_BASE, 2'd1, 4);
        write_default_descriptor(DESC_BASE, 7'd2);

        // S04 error response (DMA error) held 3 cycles.
        scenarios++;
        set_desc(DESC_BASE, 0, 'h100, 'h200, 'h0f00); // out region leaves memory
        run_cmd(GOOD_INSTR, DESC_BASE, 2'd2, 3);
        write_default_descriptor(DESC_BASE, 7'd2);

        // S05 request held across a pending response; must not be lost.
        scenarios++;
        begin
            int c;
            while (!nice_req_ready) @(negedge clk);
            @(negedge clk);
            nice_req_valid = 1; nice_req_instr = GOOD_INSTR; nice_req_rs1 = DESC_BASE;
            @(negedge clk);
            nice_req_valid = 0;
            while (!nice_rsp_valid) @(negedge clk);
            // Present the next request while the first response is pending.
            @(negedge clk);
            nice_req_valid = 1; nice_req_instr = GOOD_INSTR; nice_req_rs1 = DESC_BASE;
            assertions++;
            if (nice_req_ready !== 1'b0) $fatal(1, "REQ_READY_ASSERTED_DURING_RESPONSE");
            nice_rsp_ready = 1;
            @(negedge clk);
            nice_rsp_ready = 0;
            c = 0;
            for (c = 0; !nice_req_ready && c <= 1000; c = c + 1) @(negedge clk);
            if (!nice_req_ready) $fatal(1, "HELD_REQUEST_LOST");
            @(negedge clk);
            nice_req_valid = 0; // accepted on the preceding edge
            commands++;
            for (c = 0; !nice_rsp_valid && c <= 40000; c = c + 1) @(negedge clk);
            if (!nice_rsp_valid) $fatal(1, "TIMEOUT on held request");
            assertions++;
            if (nice_rsp_data !== 32'h0 || nice_rsp_err !== 1'b0)
                $fatal(1, "RSP_VALUE_MISMATCH held request data=%h err=%b", nice_rsp_data, nice_rsp_err);
            nice_rsp_ready = 1;
            @(negedge clk);
            nice_rsp_ready = 0;
            check_output(8'h7f, 16'h7fff);
        end

        // S06 full transaction under memory-side request backpressure.
        scenarios++;
        bp_enable = 1;
        run_cmd(GOOD_INSTR, DESC_BASE, 2'd0, 2);
        check_output(8'h7f, 16'h7fff);
        bp_enable = 0;

        // S07 last legal descriptor address: rs1 = EXT_MEM_BYTES - 32.
        scenarios++;
        write_default_descriptor(EXT_MEM_BYTES - 32, 7'd2);
        run_cmd(GOOD_INSTR, EXT_MEM_BYTES - 32, 2'd0, 1);
        check_output(8'h7f, 16'h7fff);

        // S08..S11 instruction / descriptor-address rejects (status 3).
        scenarios++;
        run_cmd(32'h0000_600a, DESC_BASE, 2'd3, 1);      // wrong opcode
        scenarios++;
        run_cmd(32'h0000_100b, DESC_BASE, 2'd3, 1);      // wrong funct3
        scenarios++;
        run_cmd(32'h0200_600b, DESC_BASE, 2'd3, 1);      // nonzero funct7
        scenarios++;
        run_cmd(GOOD_INSTR, 'hfe4, 2'd3, 1);             // misaligned rs1
        scenarios++;
        run_cmd(GOOD_INSTR, 'hff0, 2'd3, 1);             // rs1 + 32 > EXT_MEM_BYTES

        // S12..S17 illegal tile command fields (status 1).
        scenarios++;
        set_desc2(DESC_BASE, 7'd0, 1'b1, 8'h7f, 16'h7fff); run_cmd(GOOD_INSTR, DESC_BASE, 2'd1, 1);
        scenarios++;
        set_desc2(DESC_BASE, 7'd65, 1'b1, 8'h7f, 16'h7fff); run_cmd(GOOD_INSTR, DESC_BASE, 2'd1, 1);
        scenarios++;
        set_desc2(DESC_BASE, 7'd2, 1'b1, 8'h00, 16'h7fff); run_cmd(GOOD_INSTR, DESC_BASE, 2'd1, 1);
        scenarios++;
        set_desc2(DESC_BASE, 7'd2, 1'b1, 8'h7f, 16'h0000); run_cmd(GOOD_INSTR, DESC_BASE, 2'd1, 1);
        scenarios++;
        set_desc(DESC_BASE, 32'h1, 'h100, 'h200, OUT_BASE); run_cmd(GOOD_INSTR, DESC_BASE, 2'd1, 1);
        scenarios++;
        set_desc(DESC_BASE, 0, 32'h4, 'h200, OUT_BASE); run_cmd(GOOD_INSTR, DESC_BASE, 2'd1, 1);
        write_default_descriptor(DESC_BASE, 7'd2);

        // S18..S21 out-of-range transfers reported by the DMA (status 2).
        scenarios++;
        set_desc(DESC_BASE, 0, 'h100, 'h200, 'h0f00);  run_cmd(GOOD_INSTR, DESC_BASE, 2'd2, 1); // out
        scenarios++;
        set_desc(DESC_BASE, 'h0ff0, 'h100, 'h200, OUT_BASE); run_cmd(GOOD_INSTR, DESC_BASE, 2'd2, 1); // A
        scenarios++;
        set_desc(DESC_BASE, 0, 'h0fe0, 'h200, OUT_BASE); run_cmd(GOOD_INSTR, DESC_BASE, 2'd2, 1); // B
        scenarios++;
        set_desc(DESC_BASE, 0, 'h100, 'h0fe0, OUT_BASE); run_cmd(GOOD_INSTR, DESC_BASE, 2'd2, 1); // bias
        write_default_descriptor(DESC_BASE, 7'd2);

        // S22 reset in the middle of an accepted transaction, then recover.
        scenarios++;
        begin
            while (!nice_req_ready) @(negedge clk);
            @(negedge clk);
            nice_req_valid = 1; nice_req_instr = GOOD_INSTR; nice_req_rs1 = DESC_BASE;
            @(negedge clk);
            nice_req_valid = 0;
            repeat (6) @(negedge clk); // progress into load / compute
            rst_n = 0;
            repeat (3) @(negedge clk);
            assertions++;
            if (nice_rsp_valid !== 1'b0) $fatal(1, "RESPONSE_ACTIVE_DURING_MID_TXN_RESET");
            rst_n = 1;
            repeat (3) @(negedge clk);
            assertions++;
            if (nice_req_ready !== 1'b1) $fatal(1, "NOT_READY_AFTER_MID_TXN_RESET");
            run_cmd(GOOD_INSTR, DESC_BASE, 2'd0, 1);
            check_output(8'h7f, 16'h7fff);
        end

        // S23 three back-to-back commands (continuous tasks).
        scenarios++;
        run_cmd(GOOD_INSTR, DESC_BASE, 2'd0, 0);
        run_cmd(GOOD_INSTR, DESC_BASE, 2'd0, 0);
        run_cmd(GOOD_INSTR, DESC_BASE, 2'd0, 0);
        check_output(8'h7f, 16'h7fff);

        $display("PASS nice_protocol_boundaries: scenarios=%0d commands=%0d assertions=%0d bus_beats=%0d",
                 scenarios, commands, assertions, bus_beats);
        $finish;
    end

    initial begin
        #10_000_000;
        $fatal(1, "GLOBAL_TIMEOUT");
    end

endmodule
