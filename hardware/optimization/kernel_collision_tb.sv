module kernel_collision_tb;
    localparam int ROWS = 2;
    localparam int COLS = 3;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0;
    logic clear = 0;
    logic launch = 0;
    logic [ROWS*16-1:0] a_data;
    logic [COLS*16-1:0] b_data;
    logic [ROWS*COLS*32-1:0] product_data;
    logic [ROWS*COLS-1:0] product_valid;
    logic collision_seen;
    logic dump_req = 0;
    logic dump_row = 0;
    logic [1:0] dump_col = 0;
    logic dump_valid;
    logic [31:0] dump_data;

    kwt_product_array #(.ROWS(ROWS), .COLS(COLS)) product (
        .clk(clk), .rst_n(rst_n), .launch(launch),
        .a_data(a_data), .a_mask('1), .b_data(b_data), .b_mask('1),
        .product_data(product_data), .product_valid(product_valid)
    );
    kwt_cacc #(.ROWS(ROWS), .COLS(COLS)) cacc (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .product_data(product_data), .product_valid(product_valid),
        .collision_seen(collision_seen), .dump_req(dump_req),
        .dump_row(dump_row), .dump_col(dump_col),
        .dump_valid(dump_valid), .dump_data(dump_data)
    );

    task automatic issue(input int a0, a1, b0, b1, b2);
        @(negedge clk);
        launch = 1;
        a_data = {16'(a1),16'(a0)};
        b_data = {16'(b2),16'(b1),16'(b0)};
        @(negedge clk);
        launch = 0;
    endtask

    task automatic check_sum(input int r, c, expected);
        @(negedge clk);
        dump_req = 1;
        dump_row = 1'(r);
        dump_col = 2'(c);
        @(negedge clk);
        dump_req = 0;
        if (!dump_valid || dump_data !== 32'(expected))
            $fatal(1, "sum[%0d,%0d] got=%0d expected=%0d", r, c, dump_data, expected);
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        issue(1, 2, 3, 4, 5);
        repeat (COLS-2) @(negedge clk);
        issue(6, 7, 8, 9, 10);
        repeat (10) @(negedge clk);
        if (collision_seen) $fatal(1, "spaced launches collided");
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                check_sum(r, c, (r == 0 ? 1 : 2)*(c+3) + (r == 0 ? 6 : 7)*(c+8));
        $display("PASS spaced launches: CACC exact, no collision");

        @(negedge clk);
        rst_n = 0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        issue(1, 2, 3, 4, 5);
        issue(6, 7, 8, 9, 10);
        repeat (10) @(negedge clk);
        if (!collision_seen) $fatal(1, "consecutive launches did not expose collision");
        @(negedge clk);
        dump_req = 1;
        dump_row = 0;
        dump_col = 0;
        @(negedge clk);
        dump_req = 0;
        if (!dump_valid || dump_data === 32'd51)
            $fatal(1, "expected dropped row0,col0 product after collision");
        $display("PASS consecutive launches: collision drops product, row0,col0=%0d expected=51", dump_data);
        $finish;
    end
endmodule
