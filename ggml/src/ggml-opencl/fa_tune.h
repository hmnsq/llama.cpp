#pragma once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
கட்டமைப்பு ggml_opencl_fa_dim {
    முழு dk; முழு dv; முழு bm; முழு bn; முழு n_split; முழு nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
நிலையான மாறிலி ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

கட்டமைப்பு ggml_opencl_fa_dim_table {
    மாறிலி ggml_opencl_fa_dim * data;
    size_t                     மொத்தம்;

    மாறிலி ggml_opencl_fa_dim * begin() மாறிலி { திரும்பு data; }
    மாறிலி ggml_opencl_fa_dim * end()   மாறிலி { திரும்பு data + மொத்தம்; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
நிலையான ggml_opencl_fa_dim g_fa_dims_runtime[
    அளவு(g_fa_dims_adreno_default) / அளவு(g_fa_dims_adreno_default[0])];

நிலையான ggml_opencl_fa_dim_table g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    அளவு(g_fa_dims_adreno_default) / அளவு(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
நிலையான வெறுமை ggml_opencl_fa_apply_env_overrides() {
    மாறிலி எழுத்து * e = std::getenv("GGML_OPENCL_FA_TUNE");
    எனில் (!e || !e[0]) {
        திரும்பு;
    }

    std::string s = e;
    size_t pos = 0;
    போது (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        முழு dk, dv, bm, bn, nsplit, thr;
        எனில் (std::sscanf(entry.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            க்கு (size_t i = 0; i < g_opencl_fa_dims.மொத்தம்; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                எனில் (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_split = nsplit; d.nkv_split_threshold = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    நிறுத்து;
                }
            }
            எனில் (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } இல்லையெனில் {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", entry.c_str());
        }
        எனில் (comma == std::string::npos) நிறுத்து;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
நிலையான வெறுமை ggml_cl_init_fa_dims_table() {
    மாறிலி size_t மொத்தம் = அளவு(g_fa_dims_adreno_default) / அளவு(g_fa_dims_adreno_default[0]);
    க்கு (size_t i = 0; i < மொத்தம்; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, மொத்தம் };
    ggml_opencl_fa_apply_env_overrides();
}
