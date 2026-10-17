// One output tile using ping-pong A/B pages, a product-only pulse array, and
// row-banked centralized accumulation. The idle page can be loaded while the
// other page is being computed.
module kwt_pingpong_cacc_engine #(
    parameter int ROWS = 8,
    parameter int COLS = 16,
    parameter int DATA_W = 16,
    parameter int ACC_W = 32,
    parameter int SRAM_DEPTH = 1024,
    parameter int LANE_W = ($clog2(ROWS) > $clog2(COLS)) ? $clog2(ROWS) : $clog2(COLS),
    parameter int OUT_ADDR_W = 10,
    parameter int ROW_W = (ROWS <= 2) ? 1 : $clog2(ROWS),
    parameter int COL_W = (COLS <= 2) ? 1 : $clog2(COLS),
    parameter int TILE_W = (ROWS*COLS <= 2) ? 1 : $clog2(ROWS*COLS),
    parameter int OUT_DEPTH = 1024,
    parameter int OUT_INDEX_W = (OUT_DEPTH <= 2) ? 1 : $clog2(OUT_DEPTH)
) (
    input logic clk, rst_n,
    input logic start,
    input logic compute_page,
    input logic [6:0] k_len,
    input logic [ROWS-1:0] row_mask,
    input logic [COLS-1:0] col_mask,
    input logic [COLS*ACC_W-1:0] bias_data,
    input logic relu_en,
    output logic load_ready,
    output logic busy,
    output logic done,
    output logic cacc_collision,

    input logic load_page,
    input logic a_host_en, a_host_we,
    input logic [9:0] a_host_addr,
    input logic [DATA_W-1:0] a_host_wdata,
    output logic [DATA_W-1:0] a_host_rdata,
    input logic b_host_en, b_host_we,
    input logic [9:0] b_host_addr,
    input logic [DATA_W-1:0] b_host_wdata,
    output logic [DATA_W-1:0] b_host_rdata,
    input logic out_host_en, out_host_we,
    input logic [OUT_ADDR_W-1:0] out_host_addr,
    input logic [DATA_W-1:0] out_host_wdata,
    output logic [DATA_W-1:0] out_host_rdata
);
    localparam int MAX_LANES = (ROWS > COLS) ? ROWS : COLS;
    localparam int TILE_ELEMS = ROWS * COLS;
    localparam int DRAIN_LAST = ROWS + COLS + 4;

    typedef enum logic [3:0] {
        IDLE, CLEAR_CACC, READ_ISSUE, READ_CAPTURE, LAUNCH,
        DRAIN, DUMP_REQ, DUMP_WAIT, WRITE_LO, WRITE_HI, DONE
    } state_t;
    state_t state;

    logic page_q;
    logic [6:0] k_len_q, k_index;
    logic [ROWS-1:0] row_mask_q;
    logic [COLS-1:0] col_mask_q;
    logic [COLS*ACC_W-1:0] bias_q;
    logic relu_q;
    logic [LANE_W-1:0] lane_index, lane_q;
    logic [6:0] drain_count;
    logic [TILE_W-1:0] write_index;
    logic signed [DATA_W-1:0] a_lanes [0:ROWS-1];
    logic signed [DATA_W-1:0] b_lanes [0:COLS-1];
    logic [ROWS*DATA_W-1:0] a_bus;
    logic [COLS*DATA_W-1:0] b_bus;
    logic [ROWS*COLS*2*DATA_W-1:0] product_data;
    logic [ROWS*COLS-1:0] product_valid;
    logic cacc_clear, dump_req, dump_valid;
    logic [ACC_W-1:0] dump_data;
    logic [ROW_W-1:0] dump_row;
    logic [COL_W-1:0] dump_col;
    logic [DATA_W-1:0] page_a_data, page_b_data;
    logic compute_mem_en;
    logic [9:0] compute_a_addr, compute_b_addr;
    logic [ACC_W-1:0] result_q;
    logic [ACC_W:0] bias_sum;
    logic [DATA_W-1:0] out_mem [0:OUT_DEPTH-1];

    assign busy = (state != IDLE) && (state != DONE);
    assign done = (state == DONE);
    assign load_ready = !busy || (load_page != page_q);
    assign cacc_clear = (state == CLEAR_CACC);
    assign compute_mem_en = (state == READ_ISSUE);
    assign compute_a_addr = 10'(k_index * ROWS + lane_index);
    assign compute_b_addr = 10'(k_index * COLS + lane_index);
    assign dump_req = (state == DUMP_REQ);
    assign dump_row = ROW_W'(write_index / COLS);
    assign dump_col = COL_W'(write_index % COLS);
    assign bias_sum = {dump_data[ACC_W-1],dump_data} +
                      {bias_q[dump_col*ACC_W + ACC_W-1],
                       bias_q[dump_col*ACC_W +: ACC_W]};
    assign a_host_rdata = '0;
    assign b_host_rdata = '0;

    for (genvar r = 0; r < ROWS; r++) begin : g_pack_a
        assign a_bus[r*DATA_W +: DATA_W] = a_lanes[r];
    end
    for (genvar c = 0; c < COLS; c++) begin : g_pack_b
        assign b_bus[c*DATA_W +: DATA_W] = b_lanes[c];
    end

    kwt_operand_pingpong #(.DATA_W(DATA_W), .DEPTH(SRAM_DEPTH)) operands (
        .clk(clk), .rst_n(rst_n),
        .load_page(load_page), .load_ready(load_ready),
        .a_host_en(a_host_en), .a_host_we(a_host_we),
        .a_host_addr(a_host_addr), .a_host_wdata(a_host_wdata),
        .b_host_en(b_host_en), .b_host_we(b_host_we),
        .b_host_addr(b_host_addr), .b_host_wdata(b_host_wdata),
        .compute_en(compute_mem_en), .compute_page(page_q),
        .compute_a_addr(compute_a_addr), .compute_b_addr(compute_b_addr),
        .compute_a_data(page_a_data), .compute_b_data(page_b_data)
    );

    kwt_product_array #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W)) array (
        .clk(clk), .rst_n(rst_n), .launch(state == LAUNCH),
        .a_data(a_bus), .a_mask(row_mask_q),
        .b_data(b_bus), .b_mask(col_mask_q),
        .product_data(product_data), .product_valid(product_valid)
    );

    kwt_cacc #(.ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)) cacc (
        .clk(clk), .rst_n(rst_n), .clear(cacc_clear),
        .product_data(product_data), .product_valid(product_valid),
        .collision_seen(cacc_collision),
        .dump_req(dump_req), .dump_row(dump_row), .dump_col(dump_col),
        .dump_valid(dump_valid), .dump_data(dump_data)
    );

    // Output SRAM is 16-bit; INT32 results occupy two adjacent words.
    always_ff @(posedge clk) begin
        if ((state == WRITE_LO) || (state == WRITE_HI)) begin
            out_mem[OUT_INDEX_W'(write_index*2 + (state == WRITE_HI))] <=
                (row_mask_q[dump_row] && col_mask_q[dump_col]) ?
                    ((state == WRITE_HI) ? result_q[ACC_W-1:DATA_W] : result_q[DATA_W-1:0]) : '0;
        end else if (!busy && out_host_en) begin
            if (out_host_we) out_mem[OUT_INDEX_W'(out_host_addr)] <= out_host_wdata;
            else out_host_rdata <= out_mem[OUT_INDEX_W'(out_host_addr)];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            page_q <= 1'b0;
            k_len_q <= '0;
            k_index <= '0;
            row_mask_q <= '0;
            col_mask_q <= '0;
            bias_q <= '0;
            relu_q <= 1'b0;
            lane_index <= '0;
            lane_q <= '0;
            drain_count <= '0;
            write_index <= '0;
            result_q <= '0;
            for (int r = 0; r < ROWS; r++) a_lanes[r] <= '0;
            for (int c = 0; c < COLS; c++) b_lanes[c] <= '0;
        end else begin
            case (state)
                IDLE: if (start) begin
                    page_q <= compute_page;
                    k_len_q <= k_len;
                    row_mask_q <= row_mask;
                    col_mask_q <= col_mask;
                    bias_q <= bias_data;
                    relu_q <= relu_en;
                    k_index <= '0;
                    state <= CLEAR_CACC;
                end
                CLEAR_CACC: begin
                    lane_index <= '0;
                    write_index <= '0;
                    if (k_len_q == 0) state <= DUMP_REQ;
                    else state <= READ_ISSUE;
                end
                READ_ISSUE: begin
                    // The synchronous SRAM returns the previous lane while
                    // this cycle issues the next address.
                    if (lane_index != '0) begin
                        if (int'(lane_q) < ROWS) a_lanes[int'(lane_q)] <= $signed(page_a_data);
                        if (int'(lane_q) < COLS) b_lanes[int'(lane_q)] <= $signed(page_b_data);
                    end
                    lane_q <= lane_index;
                    if (lane_index == LANE_W'(MAX_LANES-1)) state <= READ_CAPTURE;
                    else lane_index <= lane_index + 1'b1;
                end
                READ_CAPTURE: begin
                    if (int'(lane_q) < ROWS) a_lanes[int'(lane_q)] <= $signed(page_a_data);
                    if (int'(lane_q) < COLS) b_lanes[int'(lane_q)] <= $signed(page_b_data);
                    state <= LAUNCH;
                end
                LAUNCH: begin
                    if (k_index + 1'b1 >= k_len_q) begin
                        drain_count <= '0;
                        state <= DRAIN;
                    end else begin
                        k_index <= k_index + 1'b1;
                        lane_index <= '0;
                        state <= READ_ISSUE;
                    end
                end
                DRAIN: begin
                    if (drain_count == 7'(DRAIN_LAST)) begin
                        write_index <= '0;
                        state <= DUMP_REQ;
                    end else drain_count <= drain_count + 1'b1;
                end
                DUMP_REQ: state <= DUMP_WAIT;
                DUMP_WAIT: if (dump_valid) begin
                    if (row_mask_q[dump_row] && col_mask_q[dump_col]) begin
                        if (relu_q && bias_sum[ACC_W-1]) result_q <= '0;
                        else result_q <= bias_sum[ACC_W-1:0];
                    end else result_q <= '0;
                    state <= WRITE_LO;
                end
                WRITE_LO: state <= WRITE_HI;
                WRITE_HI: begin
                    if (write_index == TILE_W'(TILE_ELEMS-1)) state <= DONE;
                    else begin
                        write_index <= write_index + 1'b1;
                        state <= DUMP_REQ;
                    end
                end
                DONE: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
