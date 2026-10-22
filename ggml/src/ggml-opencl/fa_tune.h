#pragma once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
સંરચના ggml_opencl_fa_dim {
    પૂર્ણાંક dk; પૂર્ણાંક dv; પૂર્ણાંક bm; પૂર્ણાંક bn; પૂર્ણાંક n_split; પૂર્ણાંક nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
સ્થિર અચલ ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

સંરચના ggml_opencl_fa_dim_table {
    અચલ ggml_opencl_fa_dim * data;
    size_t                     ગણતરી;

    અચલ ggml_opencl_fa_dim * begin() અચલ { પાછા data; }
    અચલ ggml_opencl_fa_dim * end()   અચલ { પાછા data + ગણતરી; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
સ્થિર ggml_opencl_fa_dim g_fa_dims_runtime[
    કદ(g_fa_dims_adreno_default) / કદ(g_fa_dims_adreno_default[0])];

સ્થિર ggml_opencl_fa_dim_table g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    કદ(g_fa_dims_adreno_default) / કદ(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
સ્થિર શૂન્ય ggml_opencl_fa_apply_env_overrides() {
    અચલ અક્ષર * e = std::getenv("GGML_OPENCL_FA_TUNE");
    જો (!e || !e[0]) {
        પાછા;
    }

    std::string s = e;
    size_t pos = 0;
    જ્યારે (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        પૂર્ણાંક dk, dv, bm, bn, nsplit, thr;
        જો (std::sscanf(entry.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            માટે (size_t i = 0; i < g_opencl_fa_dims.ગણતરી; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                જો (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_split = nsplit; d.nkv_split_threshold = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    તોડો;
                }
            }
            જો (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } નહીંતર {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", entry.c_str());
        }
        જો (comma == std::string::npos) તોડો;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
સ્થિર શૂન્ય ggml_cl_init_fa_dims_table() {
    અચલ size_t ગણતરી = કદ(g_fa_dims_adreno_default) / કદ(g_fa_dims_adreno_default[0]);
    માટે (size_t i = 0; i < ગણતરી; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, ગણતરી };
    ggml_opencl_fa_apply_env_overrides();
}
