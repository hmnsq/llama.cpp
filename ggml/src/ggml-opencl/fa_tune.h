#సూచన once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
నిర్మాణం ggml_opencl_fa_dim {
    పూర్ణాంకం dk; పూర్ణాంకం dv; పూర్ణాంకం bm; పూర్ణాంకం bn; పూర్ణాంకం n_విభజించు; పూర్ణాంకం nkv_విభజించు_పరిమితి;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
స్థిర స్థిరరాశి ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

నిర్మాణం ggml_opencl_fa_dim_పట్టిక {
    స్థిరరాశి ggml_opencl_fa_dim * దత్తాంశం;
    size_t                     లెక్క;

    స్థిరరాశి ggml_opencl_fa_dim * begin() స్థిరరాశి { తిరిగి దత్తాంశం; }
    స్థిరరాశి ggml_opencl_fa_dim * చివర()   స్థిరరాశి { తిరిగి దత్తాంశం + లెక్క; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
స్థిర ggml_opencl_fa_dim g_fa_dims_runtime[
    పరిమాణం(g_fa_dims_adreno_default) / పరిమాణం(g_fa_dims_adreno_default[0])];

స్థిర ggml_opencl_fa_dim_పట్టిక g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    పరిమాణం(g_fa_dims_adreno_default) / పరిమాణం(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
స్థిర శూన్య ggml_opencl_fa_apply_env_overrides() {
    స్థిరరాశి అక్షరం * e = std::getenv("GGML_OPENCL_FA_TUNE");
    ఒకవేళ (!e || !e[0]) {
        తిరిగి;
    }

    std::string s = e;
    size_t pos = 0;
    అయితే (pos < s.కొలత()) {
        size_t comma = s.కనుగొను(',', pos);
        std::string నమోదు_2 = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        పూర్ణాంకం dk, dv, bm, bn, nsplit, thr;
        ఒకవేళ (std::అక్షరశ్రేణి_చదువు(నమోదు_2.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            కొరకు (size_t i = 0; i < g_opencl_fa_dims.లెక్క; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                ఒకవేళ (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_విభజించు = nsplit; d.nkv_విభజించు_పరిమితి = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    ఆపు;
                }
            }
            ఒకవేళ (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } లేకపోతే {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", నమోదు_2.c_str());
        }
        ఒకవేళ (comma == std::string::npos) ఆపు;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
స్థిర శూన్య ggml_cl_init_fa_dims_పట్టిక() {
    స్థిరరాశి size_t లెక్క = పరిమాణం(g_fa_dims_adreno_default) / పరిమాణం(g_fa_dims_adreno_default[0]);
    కొరకు (size_t i = 0; i < లెక్క; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, లెక్క };
    ggml_opencl_fa_apply_env_overrides();
}
