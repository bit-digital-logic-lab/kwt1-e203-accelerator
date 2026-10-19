// PROPOSED coherent attention constants (r2_scale_contract).
// Frozen on the 240-sample TRAINING split (gradient-free); NOT yet
// integrated into third_party/ and NOT yet RTL-reverified for the
// full encoder chain.
localparam int ATT_T  = 99;
localparam int ATT_D  = 64;
localparam int F_BITS = 6;
localparam int M_LUT  = 14;
localparam int G_SH   = 20;
localparam int KC    = 5152;
localparam int SH_KC = 34;
localparam int RQ    = 5;
localparam int RK = 7;
localparam int RV    = 6;
localparam int RC = 15;
localparam int RO    = 11;
