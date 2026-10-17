// Capture one ROWS x COLS CACC tile into a single-port output SRAM, then
// stream its valid results with backpressure. The compute core can start its
// next tile while this buffer is being drained. The next capture waits until
// the buffer is empty. Bias is supplied by the surrounding task controller.
module kwt_tile_output_stage #(
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter int COL_W = (COLS <= 2) ? 1 : $clog2(COLS),
    parameter int TILE_W = $clog2(ROWS*COLS)
) (
    input  logic clk, rst_n,
    input  logic start_valid,
    output logic start_ready,
    input  logic [6:0] base_row,
    input  logic [7:0] base_col,
    input  logic [6:0] m_len,
    input  logic [8:0] n_len,
    output logic [7:0] bias_col,
    // One-cycle synchronous bias read: bias_col is an address in this cycle,
    // bias_data is its registered result in the following cycle.
    input  logic [31:0] bias_data,
    output logic dump_req,
    output logic [ROW_W-1:0] dump_row,
    output logic [COL_W-1:0] dump_col,
    output logic [31:0] dump_bias,
    input  logic dump_valid,
    input  logic [31:0] dump_data,
    output logic capture_done,
    output logic out_valid,
    input  logic out_ready,
    output logic [6:0] out_row,
    output logic [7:0] out_col,
    output logic [31:0] out_data,
    output logic done
);
    localparam int DEPTH = ROWS*COLS;
    typedef enum logic [2:0] {
        IDLE, BIAS_PRIME, CAPTURE, WAIT_LAST, PRIME, DRAIN
    } state_t;
    state_t state;
    logic [31:0] data_mem [0:DEPTH-1];
    logic [TILE_W-1:0] req_idx;
    logic [TILE_W:0] issued_count;
    logic [TILE_W-1:0] response_idx;
    logic response_keep;
    logic [TILE_W:0] out_count, out_idx;
    logic [6:0] base_row_q;
    logic [7:0] base_col_q;
    logic [6:0] m_len_q;
    logic [8:0] n_len_q;
    logic [COL_W:0] cols_valid;
    logic [ROW_W-1:0] local_row;
    logic [COL_W-1:0] local_col;
    logic [31:0] out_data_q;
    logic request_keep;

    assign start_ready = (state == IDLE);
    assign dump_req = (state == CAPTURE);
    assign dump_row = ROW_W'(req_idx / COLS);
    assign dump_col = COL_W'(req_idx % COLS);
    assign bias_col = base_col_q + 8'((state == CAPTURE) ?
                      ((int'(req_idx) + 1) % COLS) : (int'(req_idx) % COLS));
    assign dump_bias = bias_data;
    assign request_keep = ({1'b0, base_row_q} + 8'(dump_row) < {1'b0, m_len_q}) &&
                          ({1'b0, base_col_q} + 9'(dump_col) < n_len_q);
    assign out_valid = (state == DRAIN);
    assign out_row = base_row_q + 7'(local_row);
    assign out_col = base_col_q + 8'(local_col);
    assign out_data = out_data_q;

    // One write while capturing or one synchronous read while draining.
    // There is no read/write collision on this RAM.
    always_ff @(posedge clk) begin
        if (response_keep && dump_valid &&
            (state == CAPTURE || state == WAIT_LAST))
            data_mem[response_idx] <= dump_data;
        if (state == PRIME)
            out_data_q <= data_mem[0];
        else if (state == DRAIN && out_ready && out_idx + 1 < out_count)
            out_data_q <= data_mem[out_idx + 1];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            req_idx <= '0;
            issued_count <= '0;
            response_idx <= '0;
            response_keep <= 1'b0;
            out_count <= '0;
            out_idx <= '0;
            base_row_q <= '0;
            base_col_q <= '0;
            m_len_q <= '0;
            n_len_q <= '0;
            cols_valid <= '0;
            local_row <= '0;
            local_col <= '0;
            capture_done <= 1'b0;
            done <= 1'b0;
        end else begin
            capture_done <= 1'b0;
            done <= 1'b0;
            case (state)
                IDLE: if (start_valid) begin
                    base_row_q <= base_row;
                    base_col_q <= base_col;
                    m_len_q <= m_len;
                    n_len_q <= n_len;
                    cols_valid <= (n_len - {1'b0, base_col} >= 9'(COLS)) ?
                                  (COL_W+1)'(COLS) :
                                  (COL_W+1)'(n_len - {1'b0, base_col});
                    req_idx <= '0;
                    issued_count <= '0;
                    response_keep <= 1'b0;
                    state <= BIAS_PRIME;
                end
                BIAS_PRIME: state <= CAPTURE;
                CAPTURE: begin
                    response_keep <= request_keep;
                    response_idx <= TILE_W'(issued_count);
                    if (request_keep) issued_count <= issued_count + 1'b1;
                    if (req_idx == TILE_W'(DEPTH-1)) state <= WAIT_LAST;
                    else req_idx <= req_idx + 1'b1;
                end
                WAIT_LAST: begin
                    response_keep <= 1'b0;
                    // The core returns its last dump one cycle after request.
                    if (dump_valid) begin
                        out_count <= issued_count;
                        out_idx <= '0;
                        local_row <= '0;
                        local_col <= '0;
                        capture_done <= 1'b1;
                        if (issued_count == 0) begin
                            done <= 1'b1;
                            state <= IDLE;
                        end else state <= PRIME;
                    end
                end
                PRIME: state <= DRAIN;
                DRAIN: if (out_ready) begin
                    if (out_idx + 1 == out_count) begin
                        done <= 1'b1;
                        state <= IDLE;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                        if ({1'b0, local_col} + 1'b1 == cols_valid) begin
                            local_col <= '0;
                            local_row <= local_row + 1'b1;
                        end else local_col <= local_col + 1'b1;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
