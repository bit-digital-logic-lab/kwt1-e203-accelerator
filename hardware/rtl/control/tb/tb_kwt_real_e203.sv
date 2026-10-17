`include "e203_defines.v"

module tb_kwt_real_e203;
    logic clk = 0, rst_n = 0;
    wire [`E203_PC_SIZE-1:0] inspect_pc;
    reg [7:0] program_bytes [0:63];
    reg [8*512-1:0] program_path;
    int nice_requests = 0;
    int nice_responses = 0;
    int checked_outputs = 0;

    always #5 clk = ~clk;

    `include "e203_cpu_instance.svh"

    always @(posedge clk) begin
        if (rst_n && cpu.u_e203_cpu.u_e203_nice_core.nice_req_valid &&
            cpu.u_e203_cpu.u_e203_nice_core.nice_req_ready) begin
            nice_requests <= nice_requests + 1;
            if (cpu.u_e203_cpu.u_e203_nice_core.nice_req_rs1 !== 32'h500 ||
                cpu.u_e203_cpu.u_e203_nice_core.nice_req_inst[14:12] !== 3'b110)
                $fatal(1, "real E203 sent wrong NICE command");
        end
        if (rst_n && cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_valid &&
            cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_ready)
            nice_responses <= nice_responses + 1;
    end

    initial begin
        if (!$value$plusargs("PROGRAM=%s", program_path))
            $fatal(1, "missing +PROGRAM=byte-hex-file");
        for (int i = 0; i < 64; i++) program_bytes[i] = 0;
        $readmemh(program_path, program_bytes);
        for (int word = 0; word < 8; word++)
            for (int byte_i = 0; byte_i < 8; byte_i++)
                cpu.u_e203_srams.u_e203_itcm_ram.u_e203_itcm_gnrl_ram.u_sirv_sim_ram.mem_r[word][byte_i*8 +: 8] =
                    program_bytes[word*8+byte_i];

        for (int k = 0; k < 2; k++) begin
            for (int r = 0; r < 8; r++)
                cpu.u_e203_cpu.u_e203_nice_core.shared_ram.mem[k][r*16 +: 16] =
                    16'(k == 0 ? r+1 : -(r+1));
            for (int c = 0; c < 16; c++)
                cpu.u_e203_cpu.u_e203_nice_core.shared_ram.mem[16+k*2+c/8][(c%8)*16 +: 16] =
                    16'(c+1+k);
        end
        for (int c = 0; c < 16; c++)
            cpu.u_e203_cpu.u_e203_nice_core.shared_ram.mem[32+c/4][(c%4)*32 +: 32] = 32'(c-3);
        cpu.u_e203_cpu.u_e203_nice_core.shared_ram.mem[80] =
            {32'h300, 32'h200, 32'h100, 32'h000};
        cpu.u_e203_cpu.u_e203_nice_core.shared_ram.mem[81] =
            {64'b0, 8'b0, 16'h7fff, 8'h7f, 32'h82};

        repeat (10) @(negedge clk);
        rst_n = 1;
        for (int cycle = 0; cycle < 15000; cycle++) begin
            @(negedge clk);
            if (nice_responses == 1) begin
                if (nice_requests != 1) $fatal(1, "NICE request count=%0d", nice_requests);
                if (cpu.u_e203_cpu.u_e203_nice_core.nice_rsp_err)
                    $fatal(1, "NICE response error");
                repeat (8) @(negedge clk);
                if (cpu.u_e203_cpu.u_e203_core.u_e203_exu.u_e203_exu_regfile.rf_r[11] !== 32'b0)
                    $fatal(1, "CPU rd did not receive NICE response");
                for (int r = 0; r < 8; r++) begin
                    for (int c = 0; c < 16; c++) begin
                        int signed expected;
                        logic [31:0] actual;
                        expected = c-r-4;
                        if (expected < 0 || r == 7 || c == 15) expected = 0;
                        actual = cpu.u_e203_cpu.u_e203_nice_core.shared_ram.mem[48+(r*16+c)/4]
                                 [((r*16+c)%4)*32 +: 32];
                        if (actual !== 32'(expected))
                            $fatal(1, "output [%0d,%0d]=%h expected=%h", r, c, actual, 32'(expected));
                        checked_outputs++;
                    end
                end
                $display("PASS real E203 NICE: requests=%0d responses=%0d outputs=%0d cycles=%0d",
                         nice_requests, nice_responses, checked_outputs, cycle);
                $finish;
            end
        end
        $fatal(1, "real E203 did not finish NICE command, PC=%h requests=%0d responses=%0d",
               inspect_pc, nice_requests, nice_responses);
    end
endmodule
