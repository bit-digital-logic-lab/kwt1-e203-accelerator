// Simulation-only vector harness. The physical array remains 8 x 16.
module tb_array_vector #(
    parameter integer DATA_W = 16,
    parameter integer ACC_W = 32,
    parameter integer FORMAT = 0
);
    localparam integer ROWS = 8, COLS = 16, MAX_K = 64;
    reg clk = 0, rst_n = 0, acc_clear = 0, launch = 0;
    reg [ROWS*DATA_W-1:0] a_data = 0;
    reg [COLS*DATA_W-1:0] b_data = 0;
    reg [ROWS-1:0] a_mask = 0;
    reg [COLS-1:0] b_mask = 0;
    wire [ROWS*COLS*ACC_W-1:0] acc_data;
    wire [ROWS*COLS-1:0] acc_update;
    reg [DATA_W-1:0] a_words [0:MAX_K*ROWS-1];
    reg [DATA_W-1:0] b_words [0:MAX_K*COLS-1];
    reg [4095:0] a_path, b_path, out_path, perf_path, dump_path;
    integer k_len, rows, cols, out_fd, perf_fd;
    integer total_cycles = 0, compute_cycles = 0, wait_cycles = 0;
    integer status = 0;
    reg active = 0;

    always #5 clk = ~clk;
    kwt_pulse_array #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W),
        .ACC_W(ACC_W), .FORMAT(FORMAT)) dut (.*);

    // Counters and status belong to this harness; RTL ports are untouched.
    always @(posedge clk) if (rst_n) begin
        total_cycles <= total_cycles + 1;
        if (active) begin
            if (|acc_update) compute_cycles <= compute_cycles + 1;
            else wait_cycles <= wait_cycles + 1;
        end
    end

    initial begin
        if (!$value$plusargs("A=%s", a_path) ||
            !$value$plusargs("B=%s", b_path) ||
            !$value$plusargs("OUT=%s", out_path) ||
            !$value$plusargs("PERF=%s", perf_path) ||
            !$value$plusargs("K=%d", k_len) ||
            !$value$plusargs("ROWS=%d", rows) ||
            !$value$plusargs("COLS=%d", cols))
            $fatal(1, "missing vector argument");
        if (k_len < 1 || k_len > MAX_K || rows < 1 || rows > ROWS ||
            cols < 1 || cols > COLS) $fatal(1, "invalid shape");
        if ($value$plusargs("DUMP=%s", dump_path)) begin
            $dumpfile(dump_path);
            $dumpvars(0, tb_array_vector);
        end
        $readmemh(a_path, a_words);
        $readmemh(b_path, b_words);
        a_mask = (8'hff >> (ROWS - rows));
        b_mask = (16'hffff >> (COLS - cols));
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        acc_clear = 1;
        @(negedge clk);
        acc_clear = 0;
        active = 1;
        status = 1; // running
        for (integer k = 0; k < k_len; k = k + 1) begin
            for (integer r = 0; r < ROWS; r = r + 1)
                a_data[r*DATA_W +: DATA_W] = a_words[k*ROWS+r];
            for (integer c = 0; c < COLS; c = c + 1)
                b_data[c*DATA_W +: DATA_W] = b_words[k*COLS+c];
            launch = 1;
            @(negedge clk);
            launch = 0;
            // One accumulator per PE: drain before the next launch.
            repeat (40) @(negedge clk);
        end
        active = 0;
        status = 2; // complete
        out_fd = $fopen(out_path, "w");
        perf_fd = $fopen(perf_path, "w");
        if (!out_fd || !perf_fd) $fatal(1, "cannot open output");
        for (integer r = 0; r < rows; r = r + 1)
            for (integer c = 0; c < cols; c = c + 1)
                $fdisplay(out_fd, "%h", acc_data[(r*COLS+c)*ACC_W +: ACC_W]);
        $fdisplay(perf_fd, "status=%0d", status);
        $fdisplay(perf_fd, "total_cycles=%0d", total_cycles);
        $fdisplay(perf_fd, "compute_cycles=%0d", compute_cycles);
        $fdisplay(perf_fd, "wait_cycles=%0d", wait_cycles);
        $fdisplay(perf_fd, "dma_cycles=0");
        $fclose(out_fd);
        $fclose(perf_fd);
        $display("PASS harness shape=%0dx%0dx%0d format=%0d", rows, cols, k_len, FORMAT);
        $finish;
    end
endmodule
