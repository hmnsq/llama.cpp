#ہدایت once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
ساخت ggml_opencl_fa_dim {
    صحیح dk; صحیح dv; صحیح bm; صحیح bn; صحیح n_بانٹو; صحیح nkv_بانٹو_حد;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
ساکن ثابت ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

ساخت ggml_opencl_fa_dim_جدول {
    ثابت ggml_opencl_fa_dim * ڈیٹا;
    size_t                     گنتی;

    ثابت ggml_opencl_fa_dim * begin() ثابت { واپس ڈیٹا; }
    ثابت ggml_opencl_fa_dim * انتہا()   ثابت { واپس ڈیٹا + گنتی; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
ساکن ggml_opencl_fa_dim g_fa_dims_runtime[
    حجم(g_fa_dims_adreno_default) / حجم(g_fa_dims_adreno_default[0])];

ساکن ggml_opencl_fa_dim_جدول g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    حجم(g_fa_dims_adreno_default) / حجم(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
ساکن خالی ggml_opencl_fa_apply_env_overrides() {
    ثابت حرف * e = std::getenv("GGML_OPENCL_FA_TUNE");
    اگر (!e || !e[0]) {
        واپس;
    }

    std::string s = e;
    size_t pos = 0;
    جب (pos < s.سائز()) {
        size_t comma = s.ڈھونڈو(',', pos);
        std::string اندراج_2 = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        صحیح dk, dv, bm, bn, nsplit, thr;
        اگر (std::لڑی_سے_پڑھو(اندراج_2.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            برائے (size_t i = 0; i < g_opencl_fa_dims.گنتی; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                اگر (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_بانٹو = nsplit; d.nkv_بانٹو_حد = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    توڑو;
                }
            }
            اگر (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } ورنہ {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", اندراج_2.c_str());
        }
        اگر (comma == std::string::npos) توڑو;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
ساکن خالی ggml_cl_init_fa_dims_جدول() {
    ثابت size_t گنتی = حجم(g_fa_dims_adreno_default) / حجم(g_fa_dims_adreno_default[0]);
    برائے (size_t i = 0; i < گنتی; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, گنتی };
    ggml_opencl_fa_apply_env_overrides();
}
