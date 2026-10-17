module tb_kwt_dma;
    localparam int MEM_BYTES = 256;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0, start = 0, writeback = 0;
    logic [1:0] bank = 0;
    logic [31:0] ext_byte_addr = 0;
    logic [9:0] sram_word_addr = 0;
    logic [15:0] word_count = 0;
    logic busy, done, error;
    logic bus_valid, bus_write, bus_ready, bus_rvalid;
    logic mem_ready, throttle = 0;
    logic [31:0] bus_addr;
    logic [127:0] bus_wdata, bus_rdata;
    logic [15:0] bus_wstrb;
    logic a_en,a_we,b_en,b_we,out_en,out_we;
    logic [9:0] a_addr,b_addr,out_addr;
    logic [15:0] a_wdata,b_wdata,out_wdata,a_rdata,b_rdata,out_rdata;
    integer cycles;

    kwt_dma #(.EXT_MEM_BYTES(MEM_BYTES)) dut (.*);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) throttle <= 0;
        else throttle <= ~throttle;
    end
    assign bus_ready = mem_ready && throttle;
    kwt_ext_memory_model #(.MEM_BYTES(MEM_BYTES)) ext_mem (
        .clk, .rst_n, .valid(bus_valid && throttle), .write(bus_write), .ready(mem_ready),
        .byte_addr(bus_addr), .wdata(bus_wdata), .wstrb(bus_wstrb),
        .rvalid(bus_rvalid), .rdata(bus_rdata)
    );
    kwt_tile_buffers banks (
        .clk,
        .a_en, .a_we, .a_addr, .a_wdata, .a_rdata,
        .b_en, .b_we, .b_addr, .b_wdata, .b_rdata,
        .out_en, .out_we, .out_addr, .out_wdata, .out_rdata
    );

    task automatic transfer(input bit dir, input logic [1:0] dst_bank,
                            input int ext_addr, input int local_addr,
                            input int words, input bit expect_error);
        @(negedge clk);
        writeback = dir;
        bank = dst_bank;
        ext_byte_addr = ext_addr;
        sram_word_addr = local_addr;
        word_count = words;
        start = 1;
        @(negedge clk);
        start = 0;
        cycles = 0;
        while (!done && cycles < 200) begin
            @(negedge clk);
            cycles++;
        end
        if (!done || error !== expect_error)
            $fatal(1, "DMA transfer failed: dir=%b bank=%d ext=%d words=%d done=%b error=%b", dir,dst_bank,ext_addr,words,done,error);
        $display("DMA_CYCLES dir=%0d bank=%0d words=%0d cycles=%0d error=%0d", dir,dst_bank,words,cycles,error);
        @(negedge clk);
    endtask

    initial begin
        for (int i=0; i<MEM_BYTES/16; i++) begin
            ext_mem.mem[i] = '0;
            for (int j=0; j<8; j++)
                ext_mem.mem[i][j*16 +: 16] = 16'(i*8+j+16'h100);
        end
        repeat (3) @(negedge clk);
        rst_n = 1;
        transfer(0,0,0,5,11,0);
        for (int i=0; i<11; i++)
            if (banks.a_mem[5+i] !== 16'(16'h100+i))
                $fatal(1,"A load mismatch at %d",i);
        transfer(0,1,32,1016,8,0);
        for (int i=0; i<8; i++)
            if (banks.b_mem[1016+i] !== 16'(16'h110+i))
                $fatal(1,"B load mismatch at %d",i);
        transfer(0,2,64,7,10,0);
        for (int i=0; i<10; i++)
            if (banks.out_mem[7+i] !== 16'(16'h120+i))
                $fatal(1,"output load mismatch at %d",i);

        // 10 words span two beats. The last 6 words must retain their bytes.
        ext_mem.mem[8] = {8{16'hbeef}};
        ext_mem.mem[9] = {8{16'hbeef}};
        transfer(1,2,128,7,10,0);
        for (int i=0; i<10; i++)
            if (ext_mem.mem[8+i/8][(i%8)*16 +: 16] !== 16'(16'h120+i))
                $fatal(1,"writeback mismatch at %d",i);
        for (int i=2; i<8; i++)
            if (ext_mem.mem[9][i*16 +: 16] !== 16'hbeef)
                $fatal(1,"partial write clobbered word %d",i);

        transfer(0,2,0,100,16,0);
        transfer(1,2,192,100,16,0);
        for (int i=0; i<16; i++)
            if (ext_mem.mem[12+i/8][(i%8)*16 +: 16] !== 16'(16'h100+i))
                $fatal(1,"full-beat writeback mismatch at %d",i);
        ext_mem.mem[14] = {8{16'hbeef}};
        transfer(1,2,224,100,1,0);
        if (ext_mem.mem[14][15:0] !== 16'h100)
            $fatal(1,"single-word writeback mismatch");
        for (int i=1; i<8; i++)
            if (ext_mem.mem[14][i*16 +: 16] !== 16'hbeef)
                $fatal(1,"single-word writeback clobbered word %d",i);

        transfer(0,0,1,0,1,1);       // unaligned external address
        transfer(0,3,0,0,1,1);       // missing SRAM bank
        transfer(1,0,0,0,1,1);       // only output SRAM may write back
        transfer(0,1,0,1023,2,1);   // SRAM overflow
        transfer(0,1,240,0,9,1);    // external memory overflow
        transfer(0,1,0,0,0,1);       // zero length
        $display("PASS: DMA load/writeback, tails, bounds and invalid descriptors");
        $finish;
    end
endmodule
