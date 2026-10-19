// Byte-lane conformance testbench for KWT1-PACK v1.
//
// Reads the packed byte streams produced by the Python reference and checks
// that the RTL reconstructs every word / bit-field:
//   1. kwt_le_unpack #(16)   over the parameter pack bytes
//   2. kwt_le_unpack #(32)   over the parameter pack bytes
//   3. kwt_le_pack   #(16)   word -> bytes roundtrip
//   4. kwt_bit_unpack #(12)  PACK12 stream -> signed 12-bit values
//
// PASS requires zero mismatches and full element counts.

`timescale 1ns/1ps

module tb_byte_lane;

    localparam int N16_BYTES  = 8192;
    localparam int N16_WORDS  = N16_BYTES / 2;
    localparam int N32_BYTES  = 8192;
    localparam int N32_WORDS  = N32_BYTES / 4;
    localparam int N12_BYTES  = 9504;
    localparam int N12_VALS   = 6336;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic [7:0]  b16  [0:N16_BYTES-1];
    logic [31:0] ew16 [0:N16_WORDS-1];
    logic [7:0]  b32  [0:N32_BYTES-1];
    logic [31:0] ew32 [0:N32_WORDS-1];
    logic [7:0]  b12  [0:N12_BYTES-1];
    logic [31:0] ev12 [0:N12_VALS-1];

    // ---- DUT: 16-bit little-endian unpack --------------------------------
    logic        u16_valid, u16_out_valid;
    logic [7:0]  u16_byte;
    logic signed [31:0] u16_word;
    kwt_le_unpack #(16) dut_u16 (
        .clk(clk), .rst_n(rst_n), .in_valid(u16_valid), .in_byte(u16_byte),
        .out_valid(u16_out_valid), .out_word(u16_word));

    // ---- DUT: 32-bit little-endian unpack --------------------------------
    logic        u32_valid, u32_out_valid;
    logic [7:0]  u32_byte;
    logic signed [31:0] u32_word;
    kwt_le_unpack #(32) dut_u32 (
        .clk(clk), .rst_n(rst_n), .in_valid(u32_valid), .in_byte(u32_byte),
        .out_valid(u32_out_valid), .out_word(u32_word));

    // ---- DUT: 16-bit little-endian pack ----------------------------------
    logic               p_valid, p_ready, p_out_valid;
    logic signed [31:0] p_word;
    logic [7:0]         p_byte;
    kwt_le_pack #(16) dut_pack (
        .clk(clk), .rst_n(rst_n), .in_valid(p_valid), .in_word(p_word),
        .in_ready(p_ready), .out_valid(p_out_valid), .out_byte(p_byte));

    // ---- DUT: 12-bit LSB-first signed unpack -----------------------------
    logic        v12_valid, v12_ready, v12_out_valid;
    logic [7:0]  v12_byte;
    logic signed [31:0] v12_word;
    kwt_bit_unpack #(12) dut_v12 (
        .clk(clk), .rst_n(rst_n), .in_valid(v12_valid), .in_byte(v12_byte),
        .in_ready(v12_ready), .out_valid(v12_out_valid), .out_word(v12_word));

    // ---- scoreboards -----------------------------------------------------
    int u16_err = 0, u16_got = 0;
    int u32_err = 0, u32_got = 0;
    int v12_err = 0, v12_got = 0;
    int pk_err  = 0, pk_got  = 0;
    bit u16_done = 0, u32_done = 0, v12_done = 0, pk_done = 0;

    always @(posedge clk) begin
        if (u16_out_valid) begin
            if (u16_word !== $signed(ew16[u16_got])) u16_err <= u16_err + 1;
            u16_got <= u16_got + 1;
        end
        if (u32_out_valid) begin
            if (u32_word !== $signed(ew32[u32_got])) u32_err <= u32_err + 1;
            u32_got <= u32_got + 1;
        end
        if (v12_out_valid) begin
            if (v12_word !== $signed(ev12[v12_got])) v12_err <= v12_err + 1;
            v12_got <= v12_got + 1;
        end
        if (p_out_valid) begin
            if (p_byte !== b16[pk_got]) pk_err <= pk_err + 1;
            pk_got <= pk_got + 1;
        end
    end

    // ---- stimulus --------------------------------------------------------
    initial begin
        $readmemh("lane_bytes16.hex", b16);
        $readmemh("lane_words16.hex", ew16);
        $readmemh("lane_bytes32.hex", b32);
        $readmemh("lane_words32.hex", ew32);
        $readmemh("pack12_bytes.hex", b12);
        $readmemh("pack12_vals.hex", ev12);
    end

    // stream 16-bit bytes
    initial begin
        u16_valid = 0; u16_byte = 0;
        @(posedge rst_n);
        for (int i = 0; i < N16_BYTES; i++) begin
            @(negedge clk); u16_valid = 1; u16_byte = b16[i];
        end
        @(negedge clk); u16_valid = 0;
        repeat (4) @(negedge clk);
        u16_done = 1;
    end

    // stream 32-bit bytes
    initial begin
        u32_valid = 0; u32_byte = 0;
        @(posedge rst_n);
        for (int i = 0; i < N32_BYTES; i++) begin
            @(negedge clk); u32_valid = 1; u32_byte = b32[i];
        end
        @(negedge clk); u32_valid = 0;
        repeat (6) @(negedge clk);
        u32_done = 1;
    end

    // stream PACK12 bytes (hold each byte until the unpacker accepts it)
    initial begin : v12_drv
        int vi;
        vi = 0; v12_valid = 0; v12_byte = 0;
        @(posedge rst_n);
        @(negedge clk);
        v12_valid = 1; v12_byte = b12[0];
        while (v12_got < N12_VALS) begin
            @(negedge clk);
            if (v12_valid && v12_ready) begin
                if (vi >= N12_BYTES - 1) begin
                    v12_valid = 0;
                end else begin
                    vi = vi + 1;
                    v12_byte = b12[vi];
                    v12_valid = 1;
                end
            end
        end
        v12_valid = 0;
        repeat (4) @(negedge clk);
        v12_done = 1;
    end

    // pack words -> bytes
    initial begin : pack_drv
        int idx;
        p_valid = 0; p_word = 0; idx = 0;
        @(posedge rst_n);
        @(negedge clk); p_valid = 1; p_word = $signed(ew16[0]);
        while (pk_got < N16_BYTES) begin
            @(posedge clk);
            @(negedge clk);
            if (p_ready) begin
                if (idx >= N16_WORDS - 1) begin
                    p_valid = 0;
                end else begin
                    idx = idx + 1;
                    p_word = $signed(ew16[idx]);
                    p_valid = 1;
                end
            end
            if (idx >= N16_WORDS - 1 && pk_got >= N16_BYTES) break;
        end
        repeat (4) @(negedge clk);
        pk_done = 1;
    end

    // ---- result ----------------------------------------------------------
    int errors;
    initial begin
        rst_n = 1'b0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        while (!(u16_done && u32_done && v12_done && pk_done)) @(negedge clk);
        repeat (4) @(negedge clk);

        errors = u16_err + u32_err + v12_err + pk_err;
        $display("RTL_LANE_UNPACK16 got=%0d/%0d err=%0d", u16_got, N16_WORDS, u16_err);
        $display("RTL_LANE_UNPACK32 got=%0d/%0d err=%0d", u32_got, N32_WORDS, u32_err);
        $display("RTL_LANE_PACK12   got=%0d/%0d err=%0d", v12_got, N12_VALS, v12_err);
        $display("RTL_LANE_PACK16   got=%0d/%0d err=%0d", pk_got, N16_BYTES, pk_err);
        if (errors == 0 && u16_got == N16_WORDS && u32_got == N32_WORDS
            && v12_got == N12_VALS && pk_got == N16_BYTES) begin
            $display("RTL_LANE_PASS");
        end else begin
            $display("RTL_LANE_FAIL");
            $fatal(1, "byte-lane mismatch");
        end
        $finish;
    end

endmodule
