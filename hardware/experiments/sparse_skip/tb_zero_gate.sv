module tb_zero_gate;
    localparam int R=8, C=16, K=64;
    logic clk=0, rst_n=0, clear=0, in_valid=0, dump_req=0;
    logic [R-1:0] row_mask='0;
    logic [R*16-1:0] a_rows='0;
    logic [15:0] b_word='0;
    logic [3:0] in_col=0, dump_col=0;
    logic [2:0] dump_row=0;
    logic [31:0] dump_bias=0;
    logic v_dense,v_gate;
    logic [31:0] y_dense,y_gate;
    int cycles=0, dense_ops=0, gated_ops=0, checked=0;
    int seed=12345;
    logic signed [31:0] reference [0:R-1][0:C-1];
    always #5 clk=~clk;
    always @(posedge clk) if (rst_n) begin
        cycles <= cycles+1;
        dense_ops <= dense_ops+$countones(dense.product_valid_q);
        gated_ops <= gated_ops+$countones(gated.product_valid_q);
    end
    kwt_row_broadcast_tile #(.ROWS(R)) dense (
        .clk(clk),.rst_n(rst_n),.clear(clear),.in_valid(in_valid),.row_mask(row_mask),
        .a_rows(a_rows),.b_word(b_word),.in_col(in_col),.dump_req(dump_req),
        .dump_row(dump_row),.dump_col(dump_col),.dump_bias(dump_bias),
        .dump_valid(v_dense),.dump_data(y_dense));
    kwt_row_broadcast_tile_zero_gate #(.ROWS(R)) gated (
        .clk(clk),.rst_n(rst_n),.clear(clear),.in_valid(in_valid),.row_mask(row_mask),
        .a_rows(a_rows),.b_word(b_word),.in_col(in_col),.dump_req(dump_req),
        .dump_row(dump_row),.dump_col(dump_col),.dump_bias(dump_bias),
        .dump_valid(v_gate),.dump_data(y_gate));
    initial begin
        if ($value$plusargs("SEED=%d", seed)) begin end
        repeat (3) @(negedge clk);
        rst_n=1;
        for (int mode=0; mode<4; mode++) begin
            for (int r=0;r<R;r++) for (int c=0;c<C;c++) reference[r][c]=0;
            @(negedge clk); clear=1; in_valid=0; dump_req=0;
            @(negedge clk); clear=0; row_mask=(mode==3)?8'h1f:8'hff;
            for (int k=0;k<K;k++) begin
                for (int c=0;c<C;c++) begin
                    @(negedge clk);
                    in_valid=1; in_col=4'(c);
                    case (mode)
                        0: b_word=16'($urandom(seed));
                        1: b_word=((k+c)%4==0)?16'($urandom(seed)):0;
                        2: b_word=0;
                        3: b_word=((k+c)%3==0)?16'($urandom(seed)):0;
                    endcase
                    for (int r=0;r<R;r++) begin
                        if (mode==2 || (mode==1 && (k+r)%3!=0) ||
                            (mode==3 && (k+r)%2==0))
                            a_rows[r*16 +:16]=0;
                        else a_rows[r*16 +:16]=16'($urandom(seed));
                        if (row_mask[r]) reference[r][c] = reference[r][c] +
                            ($signed(a_rows[r*16 +:16]) * $signed(b_word));
                    end
                end
            end
            @(negedge clk); in_valid=0;
            repeat (4) @(negedge clk);
            for (int r=0;r<R;r++) for (int c=0;c<C;c++) begin
                @(negedge clk);
                dump_req=1;dump_row=3'(r);dump_col=4'(c);dump_bias=32'(r*101+c*7);
                @(negedge clk);
                if (v_dense!==1 || v_gate!==1 || y_dense!==y_gate ||
                    y_gate !== (reference[r][c]+dump_bias))
                    $fatal(1,"mode=%0d r=%0d c=%0d dense=%h gate=%h expected=%h vd=%b vg=%b",
                        mode,r,c,y_dense,y_gate,(reference[r][c]+dump_bias),v_dense,v_gate);
                checked++;
                dump_req=0;
            end
        end
        $display("PASS seed=%0d checked=%0d cycles=%0d dense_ops=%0d gated_ops=%0d",seed,checked,cycles,dense_ops,gated_ops);
        $finish;
    end
endmodule
