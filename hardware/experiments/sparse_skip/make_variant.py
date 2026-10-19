from pathlib import Path

source = Path(__file__).resolve().parents[2] / "rtl/pulse_array/kwt_row_broadcast_tile.sv"
target = Path(__file__).with_name("kwt_row_broadcast_tile_zero_gate.sv")
text = source.read_text()
text = text.replace("module kwt_row_broadcast_tile #(", "module kwt_row_broadcast_tile_zero_gate #(", 1)
text = text.replace("product_valid_q <= {ROWS{in_valid}} & row_mask;", "product_valid_q <= {ROWS{in_valid && (b_word != '0)}} & row_mask & nonzero_rows;")
text = text.replace("    logic [ROWS-1:0] product_valid_q;", "    logic [ROWS-1:0] nonzero_rows;\n    for (genvar r = 0; r < ROWS; r++) begin : g_nonzero\n        assign nonzero_rows[r] = (a_rows[r*DATA_W +: DATA_W] != '0);\n    end\n    logic [ROWS-1:0] product_valid_q;")
text = text.replace("if (in_valid) begin", "if (in_valid && (b_word != '0)) begin")
text = text.replace("                for (int r = 0; r < ROWS; r++)\n                    product_q[r] <=", "                for (int r = 0; r < ROWS; r++)\n                    if (row_mask[r] && nonzero_rows[r]) product_q[r] <=")
target.write_text(text)
print(target)
