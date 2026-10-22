#निर्देशः once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
संरचना ggml_opencl_fa_dim {
    पूर्णाङ्क dk; पूर्णाङ्क dv; पूर्णाङ्क bm; पूर्णाङ्क bn; पूर्णाङ्क n_विभज; पूर्णाङ्क nkv_विभज_देहली;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
स्थिर ध्रुव ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

संरचना ggml_opencl_fa_dim_सारणी {
    ध्रुव ggml_opencl_fa_dim * दत्त;
    size_t                     गणनम्;

    ध्रुव ggml_opencl_fa_dim * begin() ध्रुव { प्रत्यागम दत्त; }
    ध्रुव ggml_opencl_fa_dim * अन्तः_2()   ध्रुव { प्रत्यागम दत्त + गणनम्; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
स्थिर ggml_opencl_fa_dim g_fa_dims_runtime[
    परिमाण(g_fa_dims_adreno_default) / परिमाण(g_fa_dims_adreno_default[0])];

स्थिर ggml_opencl_fa_dim_सारणी g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    परिमाण(g_fa_dims_adreno_default) / परिमाण(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
स्थिर शून्य ggml_opencl_fa_apply_env_overrides() {
    ध्रुव वर्ण * e = std::getenv("GGML_OPENCL_FA_TUNE");
    यदि (!e || !e[0]) {
        प्रत्यागम;
    }

    std::string s = e;
    size_t pos = 0;
    यावत् (pos < s.आकार()) {
        size_t comma = s.find(',', pos);
        std::string प्रविष्टिः = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        पूर्णाङ्क dk, dv, bm, bn, nsplit, thr;
        यदि (std::वाक्यात्_पठ(प्रविष्टिः.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            प्रति (size_t i = 0; i < g_opencl_fa_dims.गणनम्; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                यदि (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_विभज = nsplit; d.nkv_विभज_देहली = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    भङ्ग;
                }
            }
            यदि (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } अन्यथा {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", प्रविष्टिः.c_str());
        }
        यदि (comma == std::string::npos) भङ्ग;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
स्थिर शून्य ggml_cl_init_fa_dims_सारणी() {
    ध्रुव size_t गणनम् = परिमाण(g_fa_dims_adreno_default) / परिमाण(g_fa_dims_adreno_default[0]);
    प्रति (size_t i = 0; i < गणनम्; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, गणनम् };
    ggml_opencl_fa_apply_env_overrides();
}
