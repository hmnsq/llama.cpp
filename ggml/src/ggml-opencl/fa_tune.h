#ਹਦਾਇਤ once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
ਬਣਤਰ ggml_opencl_fa_dim {
    ਪੂਰਨ dk; ਪੂਰਨ dv; ਪੂਰਨ bm; ਪੂਰਨ bn; ਪੂਰਨ n_ਵੰਡੋ; ਪੂਰਨ nkv_ਵੰਡੋ_ਹੱਦ;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
ਸਥਿਰ ਅਚਲ ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

ਬਣਤਰ ggml_opencl_fa_dim_ਸਾਰਣੀ {
    ਅਚਲ ggml_opencl_fa_dim * ਅੰਕੜੇ;
    size_t                     ਗਿਣਤੀ_2;

    ਅਚਲ ggml_opencl_fa_dim * begin() ਅਚਲ { ਵਾਪਸ ਅੰਕੜੇ; }
    ਅਚਲ ggml_opencl_fa_dim * ਸਿਰਾ()   ਅਚਲ { ਵਾਪਸ ਅੰਕੜੇ + ਗਿਣਤੀ_2; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
ਸਥਿਰ ggml_opencl_fa_dim g_fa_dims_runtime[
    ਆਕਾਰ(g_fa_dims_adreno_default) / ਆਕਾਰ(g_fa_dims_adreno_default[0])];

ਸਥਿਰ ggml_opencl_fa_dim_ਸਾਰਣੀ g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    ਆਕਾਰ(g_fa_dims_adreno_default) / ਆਕਾਰ(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
ਸਥਿਰ ਖਾਲੀ ggml_opencl_fa_apply_env_overrides() {
    ਅਚਲ ਅੱਖਰ * e = std::getenv("GGML_OPENCL_FA_TUNE");
    ਜੇ (!e || !e[0]) {
        ਵਾਪਸ;
    }

    std::string s = e;
    size_t pos = 0;
    ਜਦੋਂ (pos < s.ਮਾਪ()) {
        size_t comma = s.ਲੱਭੋ(',', pos);
        std::string ਇੰਦਰਾਜ = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        ਪੂਰਨ dk, dv, bm, bn, nsplit, thr;
        ਜੇ (std::ਲੜੀ_ਤੋਂ_ਪੜ੍ਹੋ(ਇੰਦਰਾਜ.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            ਲਈ (size_t i = 0; i < g_opencl_fa_dims.ਗਿਣਤੀ_2; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                ਜੇ (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_ਵੰਡੋ = nsplit; d.nkv_ਵੰਡੋ_ਹੱਦ = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", ਇੰਦਰਾਜ.c_str());
        }
        ਜੇ (comma == std::string::npos) ਰੋਕੋ;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
ਸਥਿਰ ਖਾਲੀ ggml_cl_init_fa_dims_ਸਾਰਣੀ() {
    ਅਚਲ size_t ਗਿਣਤੀ_2 = ਆਕਾਰ(g_fa_dims_adreno_default) / ਆਕਾਰ(g_fa_dims_adreno_default[0]);
    ਲਈ (size_t i = 0; i < ਗਿਣਤੀ_2; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, ਗਿਣਤੀ_2 };
    ggml_opencl_fa_apply_env_overrides();
}
