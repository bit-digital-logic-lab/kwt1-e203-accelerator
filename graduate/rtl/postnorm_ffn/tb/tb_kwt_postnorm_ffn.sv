// Bit-exact testbench for the KWT-1 PostNorm FFN/GELU/residual/LayerNorm fragment.
//
//   rows 0..98    real layer-1 stimulus (99 x 64, from the frozen checkpoint)
//   rows 99..107  extreme / degenerate rows (saturation, zero variance, INT32 h)
//   reset         asserted between row 98 and row 99 with output quiescence check
//   backpressure  o1_ready / y_ready come from an LFSR and stall randomly
//
// The testbench contains no golden data: it streams stimulus, dumps every
// per-stage snapshot, and fails only on X propagation or protocol violations.
// Bit-exactness is decided by model/kwt_postnorm_ref.py against RTL and against
// its own independent integer implementation.
`timescale 1ns/1ps
module tb_kwt_postnorm_ffn;
    localparam int ROWS = 109;
    localparam int RESET_AFTER = 99;
    localparam int D = 64;
    localparam int HID = 256;

    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst_n = 1'b0;

    logic x_valid, x_ready, a_valid, a_ready;
    logic signed [D*16-1:0] x_data, a_data;
    logic h_valid, h_ready;
    logic signed [31:0] h_data;
    logic o1_valid, o1_ready, y_valid, y_ready;
    logic signed [D*16-1:0] o1_data, y_data;
    logic w_we, b_we;
    logic [13:0] w_addr;
    logic signed [15:0] w_wdata;
    logic [5:0] b_addr;
    logic signed [31:0] b_wdata;
    logic dbg_r1_valid, dbg_g_valid, dbg_f_valid, dbg_r2_valid;
    logic signed [D*16-1:0] dbg_r1, dbg_r2;
    logic signed [HID*16-1:0] dbg_g;
    logic signed [D*32-1:0] dbg_f;
    logic [31:0] o1_sat_count, y_sat_count;

    kwt_postnorm_ffn dut (
        .clk, .rst_n,
        .x_valid, .x_ready, .x_data,
        .a_valid, .a_ready, .a_data,
        .h_valid, .h_ready, .h_data,
        .o1_valid, .o1_ready, .o1_data,
        .y_valid, .y_ready, .y_data,
        .w_we, .w_addr, .w_wdata,
        .b_we, .b_addr, .b_wdata,
        .dbg_r1_valid, .dbg_r1,
        .dbg_g_valid, .dbg_g,
        .dbg_f_valid, .dbg_f,
        .dbg_r2_valid, .dbg_r2,
        .o1_sat_count, .y_sat_count);

    logic [15:0] x_mem [0:ROWS*D-1];
    logic [15:0] a_mem [0:ROWS*D-1];
    logic [31:0] h_mem [0:ROWS*HID-1];
    logic [15:0] w2_mem [0:64*HID-1];
    logic [31:0] b2_mem [0:63];

    integer o1_fd, y_fd, r1_fd, g_fd, f_fd, r2_fd;
    integer o1_count = 0, y_count = 0, r1_count = 0, g_count = 0, f_count = 0, r2_count = 0;
    integer cycles = 0;
    string p_x, p_a, p_h, p_w2, p_b2, p_out;

    always_ff @(posedge clk) cycles <= cycles + 1;
    always_ff @(posedge clk)
        if (cycles % 20000 == 0 && cycles != 0)
            $display("HEARTBEAT cycles=%0d rows_done=%0d", cycles, y_count);

    // ---- random backpressure ----
    logic [31:0] lfsr = 32'h1a2b_3c4d;
    always_ff @(posedge clk) lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
    assign o1_ready = lfsr[3];
    assign y_ready  = lfsr[9];

    // ---- per-stage dumps ----
    always_ff @(posedge clk) begin
        if (o1_valid && o1_ready) begin
            if ($isunknown(o1_data)) $fatal(1, "o1_data contains X");
            for (int i = 0; i < D; i++) $fwrite(o1_fd, "%04x", o1_data[i*16 +: 16]);
            $fwrite(o1_fd, "\n");
            o1_count <= o1_count + 1;
        end
        if (y_valid && y_ready) begin
            if ($isunknown(y_data)) $fatal(1, "y_data contains X");
            for (int i = 0; i < D; i++) $fwrite(y_fd, "%04x", y_data[i*16 +: 16]);
            $fwrite(y_fd, "\n");
            y_count <= y_count + 1;
        end
        if (dbg_r1_valid) begin
            if ($isunknown(dbg_r1)) $fatal(1, "dbg_r1 contains X");
            for (int i = 0; i < D; i++) $fwrite(r1_fd, "%04x", dbg_r1[i*16 +: 16]);
            $fwrite(r1_fd, "\n");
            r1_count <= r1_count + 1;
        end
        if (dbg_g_valid) begin
            if ($isunknown(dbg_g)) $fatal(1, "dbg_g contains X");
            for (int i = 0; i < HID; i++) $fwrite(g_fd, "%04x", dbg_g[i*16 +: 16]);
            $fwrite(g_fd, "\n");
            g_count <= g_count + 1;
        end
        if (dbg_f_valid) begin
            if ($isunknown(dbg_f)) $fatal(1, "dbg_f contains X");
            for (int i = 0; i < D; i++) $fwrite(f_fd, "%08x", dbg_f[i*32 +: 32]);
            $fwrite(f_fd, "\n");
            f_count <= f_count + 1;
        end
        if (dbg_r2_valid) begin
            if ($isunknown(dbg_r2)) $fatal(1, "dbg_r2 contains X");
            for (int i = 0; i < D; i++) $fwrite(r2_fd, "%04x", dbg_r2[i*16 +: 16]);
            $fwrite(r2_fd, "\n");
            r2_count <= r2_count + 1;
        end
    end

    function automatic logic [D*16-1:0] pack_x(input int r);
        begin
            pack_x = '0;
            for (int i = 0; i < D; i++) pack_x[i*16 +: 16] = x_mem[r*D + i];
        end
    endfunction

    function automatic logic [D*16-1:0] pack_a(input int r);
        begin
            pack_a = '0;
            for (int i = 0; i < D; i++) pack_a[i*16 +: 16] = a_mem[r*D + i];
        end
    endfunction

    task automatic drive_xa(input int r);
        begin
            @(negedge clk);
            x_valid = 1'b1;
            a_valid = 1'b1;
            x_data  = pack_x(r);
            a_data  = pack_a(r);
            @(posedge clk);
            while (!x_ready) @(posedge clk);
            @(negedge clk);
            x_valid = 1'b0;
            a_valid = 1'b0;
        end
    endtask

    task automatic drive_h(input int r);
        begin
            for (int c = 0; c < HID; c++) begin
                @(negedge clk);
                h_valid = 1'b1;
                h_data  = h_mem[r*HID + c];
                @(posedge clk);
                while (!h_ready) @(posedge clk);
            end
            @(negedge clk);
            h_valid = 1'b0;
            h_data  = '0;
        end
    endtask

    task automatic load_params();
        begin
            for (int k = 0; k < 64; k++) begin
                for (int j = 0; j < HID; j++) begin
                    @(negedge clk);
                    w_we    = 1'b1;
                    w_addr  = 14'((k * HID + j) & (HID*64-1));
                    w_wdata = w2_mem[k*HID + j];
                end
            end
            @(negedge clk);
            w_we = 1'b0;
            for (int k = 0; k < 64; k++) begin
                @(negedge clk);
                b_we    = 1'b1;
                b_addr  = 6'(k);
                b_wdata = b2_mem[k];
            end
            @(negedge clk);
            b_we = 1'b0;
        end
    endtask

    task automatic reset_quiescence();
        begin
            @(negedge clk);
            rst_n = 1'b0;
            repeat (4) @(negedge clk);
            // During reset the sequencer must sit in IDLE (ready for input) with
            // every output and every per-stage snapshot deasserted.
            if (!x_ready || !a_ready)
                $fatal(1, "sequencer not idle while rst_n low");
            if (o1_valid || y_valid || dbg_r1_valid || dbg_g_valid ||
                dbg_f_valid || dbg_r2_valid)
                $fatal(1, "output asserted while rst_n low");
            rst_n = 1'b1;
            @(negedge clk);
        end
    endtask

    initial begin
        if (!$value$plusargs("XHEX=%s", p_x)) p_x = "vectors/x.hex";
        if (!$value$plusargs("AHEX=%s", p_a)) p_a = "vectors/a.hex";
        if (!$value$plusargs("HHEX=%s", p_h)) p_h = "vectors/h.hex";
        if (!$value$plusargs("W2HEX=%s", p_w2)) p_w2 = "vectors/w2.hex";
        if (!$value$plusargs("B2HEX=%s", p_b2)) p_b2 = "vectors/b2.hex";
        if (!$value$plusargs("OUT=%s", p_out)) p_out = "build";

        $readmemh(p_x, x_mem);
        $readmemh(p_a, a_mem);
        $readmemh(p_h, h_mem);
        $readmemh(p_w2, w2_mem);
        $readmemh(p_b2, b2_mem);

        o1_fd = $fopen({p_out, "/o1.hex"}, "w");
        y_fd  = $fopen({p_out, "/y.hex"}, "w");
        r1_fd = $fopen({p_out, "/r1.hex"}, "w");
        g_fd  = $fopen({p_out, "/g.hex"}, "w");
        f_fd  = $fopen({p_out, "/f.hex"}, "w");
        r2_fd = $fopen({p_out, "/r2.hex"}, "w");
        if (o1_fd == 0 || y_fd == 0 || r1_fd == 0 || g_fd == 0 || f_fd == 0 || r2_fd == 0)
            $fatal(1, "cannot open dump files under %s", p_out);

        x_valid = 0; a_valid = 0; h_valid = 0; h_data = '0;
        w_we = 0; w_wdata = '0; w_addr = '0; b_we = 0; b_wdata = '0; b_addr = '0;

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        load_params();

        for (int r = 0; r < RESET_AFTER; r++) begin
            drive_xa(r);
            drive_h(r);
        end
        while (y_count < RESET_AFTER) @(negedge clk);
        @(negedge clk);

        reset_quiescence();

        for (int r = RESET_AFTER; r < ROWS; r++) begin
            drive_xa(r);
            drive_h(r);
        end
        while (y_count < ROWS) @(negedge clk);
        repeat (4) @(negedge clk);

        $fclose(o1_fd); $fclose(y_fd); $fclose(r1_fd);
        $fclose(g_fd); $fclose(f_fd); $fclose(r2_fd);

        if (o1_count != ROWS || y_count != ROWS || r1_count != ROWS ||
            g_count != ROWS || f_count != ROWS || r2_count != ROWS)
            $fatal(1, "snapshot counts o1=%0d y=%0d r1=%0d g=%0d f=%0d r2=%0d (want %0d)",
                   o1_count, y_count, r1_count, g_count, f_count, r2_count, ROWS);

        $display("TB_PASS rows=%0d cycles=%0d o1_sat=%0d y_sat=%0d",
                 ROWS, cycles, o1_sat_count, y_sat_count);
        $finish;
    end

    initial begin
        #5000000;
        $fatal(1, "timeout");
    end
endmodule
