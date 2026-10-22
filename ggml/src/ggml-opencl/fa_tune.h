#pragma once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
ഘടന ggml_opencl_fa_dim {
    പൂർണ്ണസംഖ്യ dk; പൂർണ്ണസംഖ്യ dv; പൂർണ്ണസംഖ്യ bm; പൂർണ്ണസംഖ്യ bn; പൂർണ്ണസംഖ്യ n_split; പൂർണ്ണസംഖ്യ nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
സ്ഥിതം സ്ഥിരം ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

ഘടന ggml_opencl_fa_dim_table {
    സ്ഥിരം ggml_opencl_fa_dim * data;
    size_t                     എണ്ണം;

    സ്ഥിരം ggml_opencl_fa_dim * begin() സ്ഥിരം { തിരികെ data; }
    സ്ഥിരം ggml_opencl_fa_dim * end()   സ്ഥിരം { തിരികെ data + എണ്ണം; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
സ്ഥിതം ggml_opencl_fa_dim g_fa_dims_runtime[
    വലുപ്പം(g_fa_dims_adreno_default) / വലുപ്പം(g_fa_dims_adreno_default[0])];

സ്ഥിതം ggml_opencl_fa_dim_table g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    വലുപ്പം(g_fa_dims_adreno_default) / വലുപ്പം(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
സ്ഥിതം ശൂന്യം ggml_opencl_fa_apply_env_overrides() {
    സ്ഥിരം അക്ഷരം * e = std::getenv("GGML_OPENCL_FA_TUNE");
    എങ്കിൽ (!e || !e[0]) {
        തിരികെ;
    }

    std::string s = e;
    size_t pos = 0;
    ആകുമ്പോൾ (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        പൂർണ്ണസംഖ്യ dk, dv, bm, bn, nsplit, thr;
        എങ്കിൽ (std::sscanf(entry.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            വേണ്ടി (size_t i = 0; i < g_opencl_fa_dims.എണ്ണം; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                എങ്കിൽ (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_split = nsplit; d.nkv_split_threshold = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    നിർത്തുക;
                }
            }
            എങ്കിൽ (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } അല്ലെങ്കിൽ {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", entry.c_str());
        }
        എങ്കിൽ (comma == std::string::npos) നിർത്തുക;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
സ്ഥിതം ശൂന്യം ggml_cl_init_fa_dims_table() {
    സ്ഥിരം size_t എണ്ണം = വലുപ്പം(g_fa_dims_adreno_default) / വലുപ്പം(g_fa_dims_adreno_default[0]);
    വേണ്ടി (size_t i = 0; i < എണ്ണം; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, എണ്ണം };
    ggml_opencl_fa_apply_env_overrides();
}
