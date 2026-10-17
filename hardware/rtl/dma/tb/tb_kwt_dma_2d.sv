module tb_kwt_dma_2d;
    localparam int MEM_BYTES = 32768;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0, start = 0, writeback = 0;
    logic [1:0] bank = 0;
    logic [31:0] ext_byte_addr = 0, ext_stride_bytes = 0;
    logic [9:0] sram_word_addr = 0;
    logic [15:0] rows = 0, words_per_row = 0, sram_stride_words = 0;
    logic busy, done, error;
    logic bus_valid, bus_write, bus_ready, bus_rvalid, mem_ready;
    logic throttle = 0;
    logic [31:0] bus_addr;
    logic [127:0] bus_wdata, bus_rdata;
    logic [15:0] bus_wstrb;
    logic a_en, a_we, b_en, b_we, out_en, out_we;
    logic [9:0] a_addr, b_addr, out_addr;
    logic [15:0] a_wdata, b_wdata, out_wdata;
    logic [15:0] a_rdata, b_rdata, out_rdata;

    kwt_dma_2d #(.EXT_MEM_BYTES(MEM_BYTES)) dut (.*);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) throttle <= 0;
        else throttle <= ~throttle;
    end
    assign bus_ready = mem_ready && throttle;
    kwt_ext_memory_model #(.MEM_BYTES(MEM_BYTES)) ext_mem (
        .clk, .rst_n, .valid(bus_valid && throttle), .write(bus_write),
        .ready(mem_ready), .byte_addr(bus_addr), .wdata(bus_wdata),
        .wstrb(bus_wstrb), .rvalid(bus_rvalid), .rdata(bus_rdata)
    );
    kwt_tile_buffers banks (
        .clk,
        .a_en, .a_we, .a_addr, .a_wdata, .a_rdata,
        .b_en, .b_we, .b_addr, .b_wdata, .b_rdata,
        .out_en, .out_we, .out_addr, .out_wdata, .out_rdata
    );

    task automatic transfer(input bit dir, input logic [1:0] target_bank,
                            input int ext_base, input int local_base,
                            input int row_count, input int row_words,
                            input int ext_pitch, input int local_pitch,
                            input bit expected_error);
        int cycles;
        @(negedge clk);
        writeback = dir;
        bank = target_bank;
        ext_byte_addr = ext_base;
        sram_word_addr = 10'(local_base);
        rows = 16'(row_count);
        words_per_row = 16'(row_words);
        ext_stride_bytes = ext_pitch;
        sram_stride_words = 16'(local_pitch);
        start = 1;
        @(negedge clk);
        start = 0;
        // A live descriptor must not change the in-flight row sequence.
        ext_stride_bytes = 0;
        sram_stride_words = 0;
        rows = 0;
        words_per_row = 0;
        cycles = 0;
        while (!done && cycles < 4000) begin
            @(negedge clk);
            cycles++;
        end
        if (!done || error !== expected_error)
            $fatal(1, "2D DMA failed dir=%0d bank=%0d cycles=%0d done=%0d err=%0d",
                   dir, target_bank, cycles, done, error);
        @(negedge clk);
    endtask

    initial begin
        for (int beat = 0; beat < MEM_BYTES/16; beat++) begin
            ext_mem.mem[beat] = 0;
            for (int lane = 0; lane < 8; lane++)
                ext_mem.mem[beat][lane*16 +: 16] = 16'(beat*8 + lane);
        end
        repeat (3) @(negedge clk);
        rst_n = 1;

        // One full B[64,16] column tile fits exactly in a 1024-word bank.
        transfer(0, 1, 0, 0, 64, 16, 512, 16, 0);
        for (int row = 0; row < 64; row++)
            for (int col = 0; col < 16; col++)
                if (banks.b_mem[row*16+col] !== 16'(row*256+col))
                    $fatal(1, "B 2D mismatch row=%0d col=%0d", row, col);

        // Output tile rows are 32-bit words: 6 x 16-bit words per row.
        // External C[99,256] row stride is 1024 bytes; untouched beats stay.
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 6; col++)
                banks.out_mem[row*16+col] = 16'(16'h9000+row*16+col);
        for (int row = 0; row < 3; row++)
            ext_mem.mem[(1024+row*1024)/16] = {8{16'hbeef}};
        transfer(1, 2, 1024, 0, 3, 6, 1024, 16, 0);
        for (int row = 0; row < 3; row++) begin
            for (int col = 0; col < 6; col++)
                if (ext_mem.mem[(1024+row*1024)/16][col*16 +: 16]
                        !== 16'(16'h9000+row*16+col))
                    $fatal(1, "C 2D mismatch row=%0d col=%0d", row, col);
            for (int col = 6; col < 8; col++)
                if (ext_mem.mem[(1024+row*1024)/16][col*16 +: 16] !== 16'hbeef)
                    $fatal(1, "C tail clobbered row=%0d col=%0d", row, col);
        end

        transfer(0, 1, 0, 0, 3, 16, 510, 16, 1); // unaligned row pitch
        transfer(0, 1, 0, 1000, 3, 16, 512, 16, 1); // local overrun
        transfer(0, 1, MEM_BYTES-512, 0, 3, 16, 512, 16, 1); // external overrun
        transfer(0, 1, 0, 0, 0, 16, 512, 16, 1); // empty rows
        $display("PASS: 2D DMA B-column load, C-strided writeback, throttle, tails, bounds");
        $finish;
    end
endmodule
