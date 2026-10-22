#هدايت once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
بنية ggml_opencl_fa_dim {
    صحيح dk; صحيح dv; صحيح bm; صحيح bn; صحيح n_ورهايو; صحيح nkv_ورهايو_حد;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
ساكن ثابت ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

بنية ggml_opencl_fa_dim_جدول {
    ثابت ggml_opencl_fa_dim * معلومات;
    size_t                     ڳڻپ;

    ثابت ggml_opencl_fa_dim * begin() ثابت { أرجع معلومات; }
    ثابت ggml_opencl_fa_dim * پڇاڙي()   ثابت { أرجع معلومات + ڳڻپ; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
ساكن ggml_opencl_fa_dim g_fa_dims_runtime[
    حجم(g_fa_dims_adreno_default) / حجم(g_fa_dims_adreno_default[0])];

ساكن ggml_opencl_fa_dim_جدول g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    حجم(g_fa_dims_adreno_default) / حجم(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
ساكن فراغ ggml_opencl_fa_apply_env_overrides() {
    ثابت حرف * e = std::getenv("GGML_OPENCL_FA_TUNE");
    إذا (!e || !e[0]) {
        أرجع;
    }

    std::string s = e;
    size_t pos = 0;
    طالما (pos < s.سائيز()) {
        size_t comma = s.ڳوليو(',', pos);
        std::string داخلا = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        صحيح dk, dv, bm, bn, nsplit, thr;
        إذا (std::لڙي_مان_پڙهو(داخلا.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            لكل (size_t i = 0; i < g_opencl_fa_dims.ڳڻپ; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                إذا (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_ورهايو = nsplit; d.nkv_ورهايو_حد = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    اكسر;
                }
            }
            إذا (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } وإلا {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", داخلا.c_str());
        }
        إذا (comma == std::string::npos) اكسر;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
ساكن فراغ ggml_cl_init_fa_dims_جدول() {
    ثابت size_t ڳڻپ = حجم(g_fa_dims_adreno_default) / حجم(g_fa_dims_adreno_default[0]);
    لكل (size_t i = 0; i < ڳڻپ; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, ڳڻپ };
    ggml_opencl_fa_apply_env_overrides();
}
