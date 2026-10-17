module tb_kwt_cacc_forward;
    localparam int ROWS = 2;
    localparam int COLS = 2;
    localparam int DATA_W = 8;
    localparam int ACC_W = 16;

    logic clk = 0;
    logic rst_n = 0;
    logic clear = 0;
    logic [ROWS*COLS*2*DATA_W-1:0] product_data = '0;
    logic [ROWS*COLS-1:0] product_valid = '0;
    logic collision_seen;
    logic dump_req = 0;
    logic dump_row = 0;
    logic dump_col = 0;
    logic dump_valid;
    logic [ACC_W-1:0] dump_data;

    always #5 clk = ~clk;

    kwt_cacc #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)) dut (.*);

    task automatic send_product(input int row, input int col, input int signed value);
        @(negedge clk);
        product_valid = '0;
        product_valid[row*COLS+col] = 1'b1;
        product_data[(row*COLS+col)*2*DATA_W +: 2*DATA_W] = 16'(value);
    endtask

    task automatic check_sum(input int row, input int col, input int signed expected);
        @(negedge clk);
        dump_req = 1'b1;
        dump_row = 1'(row);
        dump_col = 1'(col);
        @(negedge clk);
        if (!dump_valid || $signed(dump_data) !== 16'(expected))
            $fatal(1, "row=%0d col=%0d got=%0d expected=%0d valid=%0b",
                   row, col, $signed(dump_data), expected, dump_valid);
        dump_req = 1'b0;
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        send_product(0, 0, 10);
        send_product(0, 0, -3);
        send_product(0, 0, 5);
        send_product(1, 1, 7);
        @(negedge clk);
        product_valid = '0;
        repeat (2) @(negedge clk);
        check_sum(0, 0, 12);
        check_sum(1, 1, 7);
        check_sum(0, 1, 0);
        if (collision_seen) $fatal(1, "unexpected row collision");

        @(negedge clk);
        clear = 1'b1;
        @(negedge clk);
        clear = 1'b0;
        check_sum(0, 0, 0);
        $display("PASS CACC forwarding: same-column back-to-back products and clear");
        $finish;
    end
endmodule
