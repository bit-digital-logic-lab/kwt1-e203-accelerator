// Freezes the *same* real A12W10 FFN1 vectors used by the legacy E203 path
// (layer01_a16/w16/bias32/expected32.hex) and drives the 16-row adapter for a
// single tile through its NICE-like command port + shared 128-bit DMA bus +
// external memory model. The expected32 file is produced independently by
// software (INT64 XW+b low 32 bits); this testbench never computes XW+b itself
// and never feeds the golden value back as DUT output.
`include "kwt_memory_defs.svh"
module tb_kwt_row16_adapter;
    localparam int M = 99, K = 64, N = 256;
    localparam int COL_TILES = 16, ROW_TILES = 7, ROWS = 16, COLS = 16;
    localparam int A_BASE = 'h1000;
    localparam int W_BASE = 'h4000;
    localparam int BIAS_BASE = 'h8000;
    localparam int OUT_BASE = 'h10000;
    localparam int EXT_MEM_BYTES = 512 * 1024;

    logic clk = 0, rst_n = 0;
    logic cmd_valid = 0, cmd_ready;
    logic [31:0] cmd_a_addr = A_BASE, cmd_b_addr = W_BASE;
    logic [31:0] cmd_bias_addr = BIAS_BASE, cmd_out_addr = OUT_BASE;
    logic [6:0] cmd_k_len = 64;
    logic [ROWS-1:0] cmd_row_mask = {ROWS{1'b1}};
    logic [COLS-1:0] cmd_col_mask = {COLS{1'b1}};
    logic cmd_relu_en = 0;
    logic rsp_valid, rsp_ready = 0;
    logic [1:0] rsp_status;
    logic bus_valid, bus_write, bus_ready;
    logic [31:0] bus_addr;
    logic [127:0] bus_wdata, bus_rdata;
    logic [15:0] bus_wstrb;
    logic bus_rvalid;

    localparam int A_TILES8 = 13;   // the frozen a16.hex is 8-row tiled
    logic [15:0] a_words [0:A_TILES8*K*8-1];
    logic [15:0] w_words [0:COL_TILES*K*16-1];
    logic [31:0] bias_words [0:N-1];
    logic [31:0] expected_words [0:M*N-1];
    string a_path, w_path, bias_path, expected_path;
    int checked_outputs = 0, checked_padding = 0;
    int cycle_count = 0;

    always #5 clk = ~clk;
    always @(posedge clk) if (rst_n) cycle_count <= cycle_count + 1;

    kwt_row16_tile_adapter #(.EXT_MEM_BYTES(EXT_MEM_BYTES), .ROWS(ROWS),
                             .COLS(COLS)) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_a_addr(cmd_a_addr), .cmd_b_addr(cmd_b_addr),
        .cmd_bias_addr(cmd_bias_addr), .cmd_out_addr(cmd_out_addr),
        .cmd_k_len(cmd_k_len), .cmd_row_mask(cmd_row_mask),
        .cmd_col_mask(cmd_col_mask), .cmd_relu_en(cmd_relu_en),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(rsp_status),
        .bus_valid(bus_valid), .bus_write(bus_write), .bus_ready(bus_ready),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_wstrb(bus_wstrb),
        .bus_rvalid(bus_rvalid), .bus_rdata(bus_rdata)
    );
    kwt_ext_memory_model #(.MEM_BYTES(EXT_MEM_BYTES)) memory (
        .clk(clk), .rst_n(rst_n), .valid(bus_valid), .write(bus_write),
        .ready(bus_ready), .byte_addr(bus_addr), .wdata(bus_wdata),
        .wstrb(bus_wstrb), .rvalid(bus_rvalid), .rdata(bus_rdata)
    );

    task automatic put16(input int byte_addr, input logic [15:0] value);
        memory.mem[byte_addr/16][((byte_addr%16)/2)*16 +: 16] = value;
    endtask
    function automatic logic [15:0] get16(input int byte_addr);
        return memory.mem[byte_addr/16][((byte_addr%16)/2)*16 +: 16];
    endfunction

    // Place one 16-row x 16-col tile into the shared external memory, derived
    // from the frozen 8-row-tiled real vectors. Layouts:
    //   A : [k][row]  -> word k*16+row
    //   W : [k][col]  -> word k*16+col
    //   bias: little-endian 32-bit at word 2*col
    task automatic setup_tile(input int row_tile, input int col_tile);
        for (int k = 0; k < K; k++) begin
            for (int r = 0; r < ROWS; r++) begin
                int row = row_tile*ROWS + r;
                logic [15:0] value = (row < M) ?
                    a_words[((row/8)*K + k)*8 + (row%8)] : 16'h0;
                put16(A_BASE + (k*ROWS + r)*2, value);
            end
        end
        for (int k = 0; k < K; k++) begin
            for (int c = 0; c < COLS; c++) begin
                int col = col_tile*COLS + c;
                put16(W_BASE + (k*COLS + c)*2,
                      w_words[((col/16)*K + k)*16 + (col%16)]);
            end
        end
        for (int c = 0; c < COLS; c++) begin
            int col = col_tile*COLS + c;
            put16(BIAS_BASE + (2*c)*2, bias_words[col][15:0]);
            put16(BIAS_BASE + (2*c+1)*2, bias_words[col][31:16]);
        end
    endtask

    task automatic submit(input logic [1:0] expected_status);
        int cycle = 0;
        logic done = 0;
        if (!cmd_ready) $fatal(1, "command interface not ready");
        @(negedge clk);
        cmd_valid = 1;
        @(negedge clk);
        cmd_valid = 0;
        while (!done) begin
            if (rsp_valid) begin
                if (rsp_status !== expected_status)
                    $fatal(1, "status=%0d expected=%0d", rsp_status, expected_status);
                rsp_ready = 1;
                @(negedge clk);
                rsp_ready = 0;
                done = 1;
            end else begin
                @(negedge clk);
                cycle++;
                if (cycle > 200000)
                    $fatal(1, "adapter timeout state=%0d", dut.state);
            end
        end
    endtask

    task automatic check_tile(input int row_tile, input int col_tile);
        for (int r = 0; r < ROWS; r++) begin
            for (int c = 0; c < COLS; c++) begin
                int row = row_tile*ROWS + r;
                int col = col_tile*COLS + c;
                int idx = r*COLS + c;
                logic [31:0] actual;
                logic [31:0] lo, hi;
                lo = get16(OUT_BASE + (2*idx)*2);
                hi = get16(OUT_BASE + (2*idx+1)*2);
                actual = lo | (hi << 16);
                if (row < M) begin
                    if (actual !== expected_words[row*N+col])
                        $fatal(1, "tile(%0d,%0d) output[%0d,%0d] got=%h expected=%h",
                               row_tile, col_tile, row, col, actual,
                               expected_words[row*N+col]);
                    checked_outputs++;
                end else begin
                    if (actual !== bias_words[col])
                        $fatal(1, "padding[%0d,%0d] got=%h expected bias=%h",
                               row, col, actual, bias_words[col]);
                    checked_padding++;
                end
            end
        end
    endtask

    initial begin : run
        if (!$value$plusargs("A=%s", a_path) ||
            !$value$plusargs("W=%s", w_path) ||
            !$value$plusargs("BIAS=%s", bias_path) ||
            !$value$plusargs("EXPECTED=%s", expected_path))
            $fatal(1, "missing +A/+W/+BIAS/+EXPECTED hex paths");
        $readmemh(a_path, a_words);
        $readmemh(w_path, w_words);
        $readmemh(bias_path, bias_words);
        $readmemh(expected_path, expected_words);
        repeat (5) @(negedge clk);
        rst_n = 1;

        // Tile 0: rows 0..15, cols 0..15 (full 16-row mask).
        setup_tile(0, 0);
        cmd_a_addr = A_BASE; cmd_b_addr = W_BASE; cmd_bias_addr = BIAS_BASE;
        cmd_out_addr = OUT_BASE; cmd_k_len = 64;
        cmd_row_mask = {ROWS{1'b1}}; cmd_col_mask = {COLS{1'b1}}; cmd_relu_en = 0;
        $display("Running 16-row tile (row_tile=0, col_tile=0)");
        submit(0);
        check_tile(0, 0);

        // Last tile: rows 96,97,98 (3 active rows, mask 0x0007), cols 240..255.
        setup_tile(ROW_TILES-1, COL_TILES-1);
        cmd_a_addr = A_BASE; cmd_b_addr = W_BASE; cmd_bias_addr = BIAS_BASE;
        cmd_out_addr = OUT_BASE; cmd_k_len = 64;
        cmd_row_mask = 16'h0007; cmd_col_mask = {COLS{1'b1}}; cmd_relu_en = 0;
        $display("Running 16-row tail tile (row_tile=6, col_tile=15, mask=0x0007)");
        submit(0);
        check_tile(ROW_TILES-1, COL_TILES-1);

        $display("PASS 16-row adapter real A12W10 tiles: outputs=%0d padding=%0d total_cycles=%0d",
                 checked_outputs, checked_padding, cycle_count);
        $finish;
    end
endmodule
