#pragma once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
sa_nracanaa ggml_opencl_fa_dim {
    ຈຳນວນເຕັມ dk; ຈຳນວນເຕັມ dv; ຈຳນວນເຕັມ bm; ຈຳນວນເຕັມ bn; ຈຳນວນເຕັມ n_split; ຈຳນວນເຕັມ nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
static const ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

sa_nracanaa ggml_opencl_fa_dim_table {
    const ggml_opencl_fa_dim * data;
    size_t                     count;

    const ggml_opencl_fa_dim * begin() const { ສົ່ງຄືນ data; }
    const ggml_opencl_fa_dim * end()   const { ສົ່ງຄືນ data + count; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
static ggml_opencl_fa_dim g_fa_dims_runtime[
    parimaa_na(g_fa_dims_adreno_default) / parimaa_na(g_fa_dims_adreno_default[0])];

static ggml_opencl_fa_dim_table g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    parimaa_na(g_fa_dims_adreno_default) / parimaa_na(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
static void ggml_opencl_fa_apply_env_overrides() {
    const var_na * e = std::getenv("GGML_OPENCL_FA_TUNE");
    ຖ້າ (!e || !e[0]) {
        ສົ່ງຄືນ;
    }

    std::string s = e;
    size_t pos = 0;
    ໃນຂະນະທີ່ (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        ຈຳນວນເຕັມ dk, dv, bm, bn, nsplit, thr;
        ຖ້າ (std::sscanf(entry.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            ສຳລັບ (size_t i = 0; i < g_opencl_fa_dims.count; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                ຖ້າ (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_split = nsplit; d.nkv_split_threshold = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    bha_nga;
                }
            }
            ຖ້າ (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } ບໍ່ດັ່ງນັ້ນ {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", entry.c_str());
        }
        ຖ້າ (comma == std::string::npos) bha_nga;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
static void ggml_cl_init_fa_dims_table() {
    const size_t count = parimaa_na(g_fa_dims_adreno_default) / parimaa_na(g_fa_dims_adreno_default[0]);
    ສຳລັບ (size_t i = 0; i < count; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, count };
    ggml_opencl_fa_apply_env_overrides();
}
