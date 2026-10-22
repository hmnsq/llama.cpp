#pragma once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
ସଂରଚନା ggml_opencl_fa_dim {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dk; ପୂର୍ଣ୍ଣସଂଖ୍ୟା dv; ପୂର୍ଣ୍ଣସଂଖ୍ୟା bm; ପୂର୍ଣ୍ଣସଂଖ୍ୟା bn; ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_split; ପୂର୍ଣ୍ଣସଂଖ୍ୟା nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
ସ୍ଥିର ଅଚଳ ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

ସଂରଚନା ggml_opencl_fa_dim_table {
    ଅଚଳ ggml_opencl_fa_dim * data;
    size_t                     ଗଣନା_2;

    ଅଚଳ ggml_opencl_fa_dim * begin() ଅଚଳ { ଫେରନ୍ତୁ data; }
    ଅଚଳ ggml_opencl_fa_dim * end()   ଅଚଳ { ଫେରନ୍ତୁ data + ଗଣନା_2; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
ସ୍ଥିର ggml_opencl_fa_dim g_fa_dims_runtime[
    ଆକାର(g_fa_dims_adreno_default) / ଆକାର(g_fa_dims_adreno_default[0])];

ସ୍ଥିର ggml_opencl_fa_dim_table g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    ଆକାର(g_fa_dims_adreno_default) / ଆକାର(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
ସ୍ଥିର ଶୂନ୍ୟ ggml_opencl_fa_apply_env_overrides() {
    ଅଚଳ ଅକ୍ଷର * e = std::getenv("GGML_OPENCL_FA_TUNE");
    ଯଦି (!e || !e[0]) {
        ଫେରନ୍ତୁ;
    }

    std::string s = e;
    size_t pos = 0;
    ଯେତେବେଳେ (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା dk, dv, bm, bn, nsplit, thr;
        ଯଦି (std::sscanf(entry.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            ପାଇଁ (size_t i = 0; i < g_opencl_fa_dims.ଗଣନା_2; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                ଯଦି (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_split = nsplit; d.nkv_split_threshold = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", entry.c_str());
        }
        ଯଦି (comma == std::string::npos) ଭାଙ୍ଗ;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
ସ୍ଥିର ଶୂନ୍ୟ ggml_cl_init_fa_dims_table() {
    ଅଚଳ size_t ଗଣନା_2 = ଆକାର(g_fa_dims_adreno_default) / ଆକାର(g_fa_dims_adreno_default[0]);
    ପାଇଁ (size_t i = 0; i < ଗଣନା_2; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, ଗଣନା_2 };
    ggml_opencl_fa_apply_env_overrides();
}
