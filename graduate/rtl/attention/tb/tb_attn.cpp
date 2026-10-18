// Verilator testbench for attn_single_head.
//
// It only (a) loads frozen inputs/parameters into the DUT load port, (b) waits
// for done, (c) reads every result memory through the verify read port and
// compares bit-for-bit with the independent integer model, and (d) computes the
// FP32 error metric from the model's float64 reference. It performs no GEMM,
// softmax, LUT or division itself.
//
// Usage: ./tb_attn <vectors_dir>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include "verilated.h"
#include "Vattn_single_head.h"

static const int T = 99, D = 64;

static std::vector<long long> load_ints(const std::string& path) {
    std::vector<long long> v;
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    long long x;
    while (f >> x) v.push_back(x);
    return v;
}

static double load_scale_o(const std::string& path) {
    std::ifstream f(path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto pos = s.find("\"scale_o\"");
    if (pos == std::string::npos) { std::cerr << "scale_o not found\n"; std::exit(2); }
    pos = s.find(':', pos);
    return std::strtod(s.c_str() + pos + 1, nullptr);
}

static Vattn_single_head* dut;
static vluint64_t cyc = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cyc++;
}

static void write_word(int sel, int row, int col, int32_t data) {
    dut->ld_sel = sel; dut->ld_row = row; dut->ld_col = col;
    dut->ld_data = data; dut->ld_we = 1;
    tick();
}
static void write_mat(int sel, const std::vector<long long>& v) {
    for (int r = 0; r < D; r++)
        for (int c = 0; c < D; c++)
            write_word(sel, r, c, (int32_t)v[r * D + c]);
}
static int32_t read_word(int sel, int row, int col) {
    dut->rd_sel = sel; dut->rd_row = row; dut->rd_col = col;
    dut->eval();
    return dut->rd_data;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string vec = (argc > 1) ? argv[1] : "vectors";
    if (!vec.empty() && vec.back() != '/') vec += "/";

    auto xin   = load_ints(vec + "x.txt");
    auto wq    = load_ints(vec + "wq.txt");
    auto wk    = load_ints(vec + "wk.txt");
    auto wv    = load_ints(vec + "wv.txt");
    auto wo    = load_ints(vec + "wo.txt");
    auto bias  = load_ints(vec + "bias.txt");
    auto eq    = load_ints(vec + "exp_q.txt");
    auto ek    = load_ints(vec + "exp_k.txt");
    auto ev    = load_ints(vec + "exp_v.txt");
    auto ec    = load_ints(vec + "exp_c.txt");
    auto eo    = load_ints(vec + "exp_o.txt");
    auto ep    = load_ints(vec + "exp_p.txt");
    auto lut   = load_ints(vec + "lut.txt");
    std::vector<double> fp32;
    { std::ifstream f(vec + "fp32_o.txt"); double z; while (f >> z) fp32.push_back(z); }
    double scale_o = load_scale_o(vec + "params.json");

    dut = new Vattn_single_head;
    dut->rst_n = 0; dut->start = 0; dut->ld_we = 0;
    dut->ld_sel = 0; dut->ld_row = 0; dut->ld_col = 0; dut->ld_data = 0;
    dut->rd_sel = 0; dut->rd_row = 0; dut->rd_col = 0;
    for (int k = 0; k < 4; k++) tick();
    dut->rst_n = 1;
    tick();

    // ---- load frozen inputs / parameters ----
    for (int r = 0; r < T; r++)
        for (int c = 0; c < D; c++)
            write_word(0, r, c, (int32_t)xin[r * D + c]);
    write_mat(1, wq);
    write_mat(2, wk);
    write_mat(3, wv);
    write_mat(4, wo);
    for (int c = 0; c < D; c++) write_word(5, 0, c, (int32_t)bias[c]);
    for (int c = 0; c < (1 << 6); c++) write_word(6, c, 0, (int32_t)lut[c]);
    dut->ld_we = 0;
    tick();

    // ---- run ----
    vluint64_t t0 = cyc;
    dut->start = 1; tick();
    dut->start = 0;
    vluint64_t guard = 0;
    while (!dut->done) {
        tick();
        if (++guard > 40000000ULL) { std::cerr << "TIMEOUT\n"; return 3; }
    }
    vluint64_t cycles = cyc - t0;

    // ---- compare ----
    struct Cmp { const char* name; int sel; int rows; int cols; const std::vector<long long>* exp; };
    std::vector<Cmp> cmps = {
        {"q", 1, T, D, &eq}, {"k", 2, T, D, &ek}, {"v", 3, T, D, &ev},
        {"p", 4, T, T, &ep}, {"c", 5, T, D, &ec}, {"o", 6, T, D, &eo},
    };
    long long total = 0, mismatch = 0;
    for (auto& c : cmps) {
        long long bad = 0; long long first = -1;
        for (int r = 0; r < c.rows; r++)
            for (int k = 0; k < c.cols; k++) {
                int32_t got = read_word(c.sel, r, k);
                long long want = (*c.exp)[r * c.cols + k];
                if (got != want) {
                    if (first < 0) first = r * c.cols + k;
                    bad++;
                }
            }
        total += (long long)c.rows * c.cols;
        mismatch += bad;
        std::printf("  %-2s %5lld/%5lld bit-exact  %s%s%lld @%lld\n",
                    c.name, (long long)c.rows * c.cols - bad, (long long)c.rows * c.cols,
                    bad ? "MISMATCH " : "OK", bad ? "first " : "", bad, first);
    }

    // ---- FP32 error metric (separate from bit-exact check) ----
    int32_t o0 = read_word(6, 0, 0); (void)o0;
    double se = 0, ref2 = 0, mx = 0;
    double row_se0 = 0, row_ref0 = 0, row_seL = 0, row_refL = 0;
    for (int r = 0; r < T; r++)
        for (int c = 0; c < D; c++) {
            int32_t oi = read_word(6, r, c);
            double dec = (double)oi * scale_o;
            double ref = fp32[r * D + c];
            double e = dec - ref;
            se += e * e; ref2 += ref * ref; mx = std::max(mx, std::fabs(e));
            if (r == 0)      { row_se0 += e * e; row_ref0 += ref * ref; }
            if (r == T - 1)  { row_seL += e * e; row_refL += ref * ref; }
        }
    double nrmse = std::sqrt(se / ref2);
    double n0 = std::sqrt(row_se0 / row_ref0);
    double nL = std::sqrt(row_seL / row_refL);

    std::printf("\n  cycles          = %llu\n", (unsigned long long)cycles);
    std::printf("  bit-exact total = %lld/%lld mismatches=%lld\n", total - mismatch, total, mismatch);
    std::printf("  FP32 NRMSE      = %.5f%%  (row0 %.5f%%, row%02d %.5f%%, maxabs %.6g)\n",
                100.0 * nrmse, 100.0 * n0, T - 1, 100.0 * nL, mx);

    bool ok = (mismatch == 0) && (nrmse < 0.03);
    std::printf("\n  RESULT: %s\n", ok ? "PASS" : "FAIL");
    delete dut;
    return ok ? 0 : 1;
}
