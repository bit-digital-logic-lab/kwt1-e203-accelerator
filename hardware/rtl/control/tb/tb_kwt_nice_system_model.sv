module tb_kwt_nice_system_model;
    logic clk = 0, rst_n = 0;
    logic nice_req_valid = 0, nice_req_ready;
    logic [31:0] nice_req_instr = 32'h0000_600b;
    logic [31:0] nice_req_rs1 = 'h500, nice_req_rs2 = 0;
    logic nice_rsp_valid, nice_rsp_ready = 0, nice_rsp_err;
    logic [31:0] nice_rsp_data;
    logic bus_valid, bus_write, bus_ready;
    logic [31:0] bus_addr;
    logic [127:0] bus_wdata, bus_rdata;
    logic [15:0] bus_wstrb;
    logic bus_rvalid;
    int desc_reads = 0;
    int checks = 0;

    always #5 clk = ~clk;
    kwt_nice_system_model #(.EXT_MEM_BYTES(2048)) dut (.*);
    kwt_ext_memory_model #(.MEM_BYTES(2048)) memory (
        .clk(clk), .rst_n(rst_n), .valid(bus_valid), .write(bus_write),
        .ready(bus_ready), .byte_addr(bus_addr), .wdata(bus_wdata),
        .wstrb(bus_wstrb), .rvalid(bus_rvalid), .rdata(bus_rdata)
    );

    always @(posedge clk) begin
        if (rst_n && bus_valid && bus_ready && !bus_write &&
            (bus_addr == 'h500 || bus_addr == 'h510)) desc_reads <= desc_reads + 1;
    end

    task automatic submit(input logic [1:0] expected_status);
        @(negedge clk);
        if (!nice_req_ready) $fatal(1, "NICE request not ready");
        nice_req_valid = 1;
        @(negedge clk);
        nice_req_valid = 0;
        for (int cycle = 0; cycle < 15000; cycle++) begin
            if (nice_rsp_valid) begin
                if (nice_rsp_data !== {30'b0, expected_status} ||
                    nice_rsp_err !== (expected_status != 0))
                    $fatal(1, "NICE response data=%h err=%b expected=%0d",
                           nice_rsp_data, nice_rsp_err, expected_status);
                // Response must remain valid under CPU backpressure.
                repeat (2) begin
                    @(negedge clk);
                    if (!nice_rsp_valid || nice_rsp_data !== {30'b0, expected_status})
                        $fatal(1, "NICE response changed before handshake");
                end
                nice_rsp_ready = 1;
                @(negedge clk);
                nice_rsp_ready = 0;
                return;
            end
            @(negedge clk);
        end
        $fatal(1, "NICE system model timeout");
    endtask

    initial begin
        // Packed A[k][r], B[k][c], 16 signed INT32 biases.
        for (int k = 0; k < 2; k++) begin
            for (int r = 0; r < 8; r++)
                memory.mem[k][r*16 +: 16] = 16'(k == 0 ? r+1 : -(r+1));
            for (int c = 0; c < 16; c++)
                memory.mem[16+k*2+c/8][(c%8)*16 +: 16] = 16'(c+1+k);
        end
        for (int c = 0; c < 16; c++)
            memory.mem[32+c/4][(c%4)*32 +: 32] = 32'(c-3);

        // 32-byte descriptor at byte 0x500. Beat 0: four data addresses.
        memory.mem[80] = {32'h300, 32'h200, 32'h100, 32'h000};
        // Beat 1: flags/k_len, masks, then reserved zero words.
        memory.mem[81] = {64'b0, 8'b0, 16'h7fff, 8'h7f, 32'h82};

        repeat (3) @(negedge clk);
        rst_n = 1;
        submit(0);
        if (desc_reads != 2) $fatal(1, "descriptor read count=%0d", desc_reads);
        for (int r = 0; r < 8; r++) begin
            for (int c = 0; c < 16; c++) begin
                int signed expected;
                logic [31:0] actual;
                expected = c-r-4;
                if (expected < 0 || r == 7 || c == 15) expected = 0;
                actual = memory.mem[48+(r*16+c)/4][((r*16+c)%4)*32 +: 32];
                if (actual !== 32'(expected))
                    $fatal(1, "output [%0d,%0d]=%h expected=%h", r, c, actual, 32'(expected));
                checks++;
            end
        end

        nice_req_instr = 0;
        submit(3);
        nice_req_instr = 32'h0000_600b;
        nice_req_rs1 = 'h7f0;
        submit(3);
        nice_req_rs1 = 'h500;
        memory.mem[81][6:0] = 0;
        submit(1);
        memory.mem[81][6:0] = 2;
        memory.mem[80][127:96] = 'h710;
        submit(2);
        $display("PASS NICE system model: %0d results, descriptor fetch, response and errors", checks);
        $finish;
    end
endmodule
