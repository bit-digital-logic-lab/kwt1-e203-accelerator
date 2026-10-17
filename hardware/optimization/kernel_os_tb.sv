module kernel_os_tb;
    localparam int ROWS = 2;
    localparam int COLS = 3;
    localparam int K = 16;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0;
    logic launch = 0;
    logic [ROWS*16-1:0] a_data;
    logic [COLS*16-1:0] b_data;
    logic [ROWS*COLS*32-1:0] product_data;
    logic [ROWS*COLS-1:0] product_valid;
    logic [ROWS*COLS*32-1:0] sum_data;
    int expected [ROWS][COLS];

    kwt_product_array #(.ROWS(ROWS), .COLS(COLS)) product (
        .clk(clk), .rst_n(rst_n), .launch(launch),
        .a_data(a_data), .a_mask('1), .b_data(b_data), .b_mask('1),
        .product_data(product_data), .product_valid(product_valid)
    );
    kernel_os_accum #(.ROWS(ROWS), .COLS(COLS)) accum (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .product_data(product_data), .product_valid(product_valid),
        .sum_data(sum_data)
    );

    initial begin
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) expected[r][c] = 0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        for (int z = 0; z < K; z++) begin
            int av0, av1, bv0, bv1, bv2;
            av0 = z - 8;
            av1 = 2*z - 13;
            bv0 = z - 5;
            bv1 = 7 - z;
            bv2 = 3*z - 19;
            launch = 1;
            a_data = {16'(av1),16'(av0)};
            b_data = {16'(bv2),16'(bv1),16'(bv0)};
            expected[0][0] += av0*bv0;
            expected[0][1] += av0*bv1;
            expected[0][2] += av0*bv2;
            expected[1][0] += av1*bv0;
            expected[1][1] += av1*bv1;
            expected[1][2] += av1*bv2;
            @(negedge clk);
        end
        launch = 0;
        repeat (ROWS+COLS+5) @(negedge clk);
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                if ($signed(sum_data[(r*COLS+c)*32 +: 32]) !== expected[r][c])
                    $fatal(1, "OS sum[%0d,%0d]=%0d expected=%0d",
                           r, c, $signed(sum_data[(r*COLS+c)*32 +: 32]), expected[r][c]);
        $display("PASS OS microkernel: %0d consecutive launches, %0d exact outputs", K, ROWS*COLS);
        $finish;
    end
endmodule
