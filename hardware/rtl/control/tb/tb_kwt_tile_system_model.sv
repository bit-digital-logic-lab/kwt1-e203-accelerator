module tb_kwt_tile_system_model;
    logic clk = 0;
    logic rst_n = 0;
    logic cmd_valid = 0, cmd_ready;
    logic [31:0] cmd_a_addr = 0, cmd_b_addr = 'h100;
    logic [31:0] cmd_bias_addr = 'h200, cmd_out_addr = 'h300;
    logic [6:0] cmd_k_len = 2;
    logic [7:0] cmd_row_mask = 8'h7f;
    logic [15:0] cmd_col_mask = 16'h7fff;
    logic cmd_relu_en = 1;
    logic rsp_valid, rsp_ready = 0;
    logic [1:0] rsp_status;
    logic bus_valid, bus_write, bus_ready;
    logic [31:0] bus_addr;
    logic [127:0] bus_wdata, bus_rdata;
    logic [15:0] bus_wstrb;
    logic bus_rvalid;
    int checks = 0;
    int bias_reads = 0;
    int cycle_count = 0;
    int last_bias_cycle = -1;

    always #5 clk = ~clk;
    kwt_tile_system_model #(.EXT_MEM_BYTES(2048)) dut (.*);
    kwt_ext_memory_model #(.MEM_BYTES(2048)) memory (
        .clk(clk), .rst_n(rst_n), .valid(bus_valid), .write(bus_write),
        .ready(bus_ready), .byte_addr(bus_addr), .wdata(bus_wdata),
        .wstrb(bus_wstrb), .rvalid(bus_rvalid), .rdata(bus_rdata)
    );

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
        if (rst_n && dut.bias_read && checks == 0) begin
            if (dut.tile.out_host_addr !== 10'(256 + bias_reads))
                $fatal(1, "bias read address mismatch at word %0d", bias_reads);
            if (bias_reads != 0 && cycle_count != last_bias_cycle + 1)
                $fatal(1, "bias read gap at word %0d", bias_reads);
            last_bias_cycle <= cycle_count;
            bias_reads <= bias_reads + 1;
        end
    end

    task automatic submit(input logic [1:0] expected_status);
        @(negedge clk);
        if (!cmd_ready) $fatal(1, "command interface not ready");
        cmd_valid = 1;
        @(negedge clk);
        cmd_valid = 0;
        for (int cycle = 0; cycle < 15000; cycle++) begin
            if (rsp_valid) begin
                if (rsp_status !== expected_status)
                    $fatal(1, "status=%0d expected=%0d", rsp_status, expected_status);
                rsp_ready = 1;
                @(negedge clk);
                rsp_ready = 0;
                return;
            end
            @(negedge clk);
        end
        $fatal(1, "system model timeout");
    endtask

    initial begin
        for (int k = 0; k < 2; k++) begin
            for (int r = 0; r < 8; r++)
                memory.mem[k][r*16 +: 16] = 16'(k == 0 ? r+1 : -(r+1));
            for (int c = 0; c < 16; c++)
                memory.mem[16 + k*2 + c/8][(c%8)*16 +: 16] = 16'(c+1+k);
        end
        for (int c = 0; c < 16; c++)
            memory.mem[32 + c/4][(c%4)*32 +: 32] = 32'(c-3);
        repeat (3) @(negedge clk);
        rst_n = 1;
        submit(0);
        if (bias_reads != 32)
            $fatal(1, "expected 32 consecutive bias reads, got %0d", bias_reads);

        for (int r = 0; r < 8; r++) begin
            for (int c = 0; c < 16; c++) begin
                int signed expected;
                logic [31:0] actual;
                expected = c-r-4;
                if (expected < 0 || r == 7 || c == 15) expected = 0;
                actual = memory.mem[48 + (r*16+c)/4][((r*16+c)%4)*32 +: 32];
                if (actual !== 32'(expected))
                    $fatal(1, "result [%0d,%0d] got=%h expected=%h", r, c, actual, 32'(expected));
                checks++;
            end
        end

        // Bad alignment is rejected before DMA; an out-of-range address is
        // reported by the DMA after the command has been accepted.
        cmd_a_addr = 1;
        submit(1);
        cmd_a_addr = 0;
        cmd_out_addr = 'h710;
        submit(2);
        $display("PASS tile system model: %0d results, command and DMA errors", checks);
        $finish;
    end
endmodule
