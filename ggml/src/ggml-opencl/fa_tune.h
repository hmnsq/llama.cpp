#ಸೂಚನೆ once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
ರಚನೆ ggml_opencl_fa_dim {
    ಪೂರ್ಣಾಂಕ dk; ಪೂರ್ಣಾಂಕ dv; ಪೂರ್ಣಾಂಕ bm; ಪೂರ್ಣಾಂಕ bn; ಪೂರ್ಣಾಂಕ n_ವಿಭಜಿಸು; ಪೂರ್ಣಾಂಕ nkv_ವಿಭಜಿಸು_ಮಿತಿ;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

ರಚನೆ ggml_opencl_fa_dim_ಕೋಷ್ಟಕ {
    ಸ್ಥಿರಾಂಕ ggml_opencl_fa_dim * ದತ್ತಾಂಶ;
    size_t                     ಎಣಿಕೆ_2;

    ಸ್ಥಿರಾಂಕ ggml_opencl_fa_dim * begin() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ; }
    ಸ್ಥಿರಾಂಕ ggml_opencl_fa_dim * ಕೊನೆ()   ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ + ಎಣಿಕೆ_2; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
ಸ್ಥಿರ ggml_opencl_fa_dim g_fa_dims_runtime[
    ಗಾತ್ರ(g_fa_dims_adreno_default) / ಗಾತ್ರ(g_fa_dims_adreno_default[0])];

ಸ್ಥಿರ ggml_opencl_fa_dim_ಕೋಷ್ಟಕ g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    ಗಾತ್ರ(g_fa_dims_adreno_default) / ಗಾತ್ರ(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_opencl_fa_apply_env_overrides() {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * e = std::getenv("GGML_OPENCL_FA_TUNE");
    ಆದರೆ (!e || !e[0]) {
        ಹಿಂತಿರುಗಿ;
    }

    std::string s = e;
    size_t pos = 0;
    ಆಗ (pos < s.ಗಾತ್ರ_2()) {
        size_t comma = s.ಕಂಡುಹಿಡಿ(',', pos);
        std::string ನಮೂದು = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        ಪೂರ್ಣಾಂಕ dk, dv, bm, bn, nsplit, thr;
        ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಓದು(ನಮೂದು.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            ಗಾಗಿ (size_t i = 0; i < g_opencl_fa_dims.ಎಣಿಕೆ_2; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                ಆದರೆ (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_ವಿಭಜಿಸು = nsplit; d.nkv_ವಿಭಜಿಸು_ಮಿತಿ = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", ನಮೂದು.c_str());
        }
        ಆದರೆ (comma == std::string::npos) ನಿಲ್ಲಿಸು;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cl_init_fa_dims_ಕೋಷ್ಟಕ() {
    ಸ್ಥಿರಾಂಕ size_t ಎಣಿಕೆ_2 = ಗಾತ್ರ(g_fa_dims_adreno_default) / ಗಾತ್ರ(g_fa_dims_adreno_default[0]);
    ಗಾಗಿ (size_t i = 0; i < ಎಣಿಕೆ_2; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, ಎಣಿಕೆ_2 };
    ggml_opencl_fa_apply_env_overrides();
}
