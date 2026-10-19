// KWT1-PACK v1 byte-lane primitives.
//
// These implement the *byte order* and *bit order* fixed by the format:
//   * little-endian word assembly / disassembly (matches RISC-V / E203)
//   * LSB-first signed bit-field extraction for the PACK10/PACK12 streams
//
// They are deliberately technology-independent and synthesizable (no arrays,
// no memories).  The testbench drives real packed bytes produced by the Python
// reference (see run_pack.py) and checks the RTL reconstruction word-for-word.

`default_nettype none

// ---------------------------------------------------------------------------
// Byte stream -> little-endian signed word (DW = 16 or 32).
// Consumes one byte per cycle while `in_valid`, emits one word (sign extended
// to 32 bits) on `out_valid`.
// ---------------------------------------------------------------------------
module kwt_le_unpack #(parameter int DW = 16) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic [7:0]         in_byte,
    output logic               out_valid,
    output logic signed [31:0] out_word
);
    localparam int NB = DW / 8;
    logic [DW-1:0] sh;
    logic [3:0]    cnt;
    logic          out_valid_r;
    logic signed [31:0] word_r;
    logic [DW-1:0] assembled;

    always_comb assembled = sh | ({{(DW-8){1'b0}}, in_byte} << (8 * cnt));
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sh <= '0; cnt <= '0; out_valid_r <= 1'b0; word_r <= '0;
        end else begin
            out_valid_r <= 1'b0;
            if (in_valid) begin
                if (cnt == 4'(NB-1)) begin
                    out_valid_r <= 1'b1;
                    word_r      <= {{(32-DW){assembled[DW-1]}}, assembled};
                    cnt         <= '0;
                    sh          <= '0;
                end else begin
                    sh  <= assembled;
                    cnt <= cnt + 4'd1;
                end
            end
        end
    end

    assign out_valid = out_valid_r;
    assign out_word  = word_r;
endmodule

// ---------------------------------------------------------------------------
// Little-endian signed word -> byte stream (DW = 16 or 32).
// Accepts one word on `in_valid`; emits NB bytes, LSB first.
// ---------------------------------------------------------------------------
module kwt_le_pack #(parameter int DW = 16) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic signed [31:0] in_word,
    output logic               in_ready,
    output logic               out_valid,
    output logic [7:0]         out_byte
);
    localparam int NB = DW / 8;
    logic [3:0]    cnt;
    logic          busy;
    logic [DW-1:0] w;
    logic          out_valid_r;
    logic [7:0]    byte_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= '0; busy <= 1'b0; out_valid_r <= 1'b0; byte_r <= '0; w <= '0;
        end else begin
            out_valid_r <= 1'b0;
            if (!busy) begin
                if (in_valid) begin
                    w           <= in_word[DW-1:0];
                    byte_r      <= in_word[7:0];
                    out_valid_r <= 1'b1;
                    if (NB == 1) begin
                        cnt <= '0;
                    end else begin
                        cnt  <= 4'd1;
                        busy <= 1'b1;
                    end
                end
            end else begin
                byte_r      <= w[8 * cnt +: 8];
                out_valid_r <= 1'b1;
                if (cnt == 4'(NB-1)) begin
                    busy <= 1'b0;
                    cnt  <= '0;
                end else begin
                    cnt <= cnt + 4'd1;
                end
            end
        end
    end

    assign out_valid = out_valid_r;
    assign out_byte  = byte_r;
    assign in_ready  = ~busy;
endmodule

// ---------------------------------------------------------------------------
// LSB-first signed bit-field unpacker for BITS = 10 or 12 (PACK10/PACK12).
// Consumes one input byte per cycle while `in_valid`; emits a signed value on
// `out_valid`.  Stream bit j is bit (j%8) of byte (j/8); two's complement in
// the low BITS bits.
// ---------------------------------------------------------------------------
module kwt_bit_unpack #(parameter int BITS = 12) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic [7:0]         in_byte,
    output logic               in_ready,
    output logic               out_valid,
    output logic signed [31:0] out_word
);
    logic [31:0] acc;
    logic [5:0]  n;
    logic        out_valid_r;
    logic signed [31:0] val_r;
    logic [31:0] mask;
    logic [31:0] u;
    logic [31:0] signbit;

    assign mask    = (32'd1 << BITS) - 32'd1;
    assign signbit = 32'd1 << (BITS - 1);
    always_comb u = acc & mask;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= '0; n <= '0; out_valid_r <= 1'b0; val_r <= '0;
        end else begin
            out_valid_r <= 1'b0;
            if (n >= 6'(BITS)) begin
                out_valid_r <= 1'b1;
                val_r       <= (u & signbit) ? ($signed(u) - (32'sd1 << BITS))
                                              : $signed(u);
                acc         <= acc >> BITS;
                n           <= n - 6'(BITS);
            end else if (in_valid) begin
                acc <= acc | ({24'd0, in_byte} << n);
                n   <= n + 6'd8;
            end
        end
    end

    assign out_valid = out_valid_r;
    assign out_word  = val_r;
    assign in_ready  = (n < 6'(BITS));
endmodule

`default_nettype wire
