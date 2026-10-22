#आशय once

// Flash-attention per-(dk,dv) tile tuning for the Adreno OpenCL backend.
// Isolated from ggml-opencl.cpp so the tuning numbers are easy to find and
// edit; the FA dispatch and kernel-compile logic stay in the main file.
// This header is a file section — it is #included exactly once, at the point
// in ggml-opencl.cpp where the ggml logging macros are already in scope.

// Per-(dk, dv) FA config; shared by dispatch and supports_op.
काष्ठा ggml_opencl_fa_dim {
    पूर्णांक dk; पूर्णांक dv; पूर्णांक bm; पूर्णांक bn; पूर्णांक n_split; पूर्णांक nkv_split_threshold;
};

// Split variant fires when n_kv >= threshold (threshold=0 -> always split).
// Default tuning covers Adreno 7xx/8xx mobile and X1-series laptop GPUs.
अटल स्थिर ggml_opencl_fa_dim g_fa_dims_adreno_default[] = {
    { 40,  40, 64, 32, 1, 0}, { 64,  64, 64, 32, 2, 64},
    { 80,  80, 64, 32, 2, 64}, { 96,  96, 64, 32, 2, 64},
    {112, 112, 64, 32, 2, 64}, {128, 128, 64, 32, 2, 64},
    {192, 128, 16, 16, 1, 0},
    {192, 192, 16, 16, 1, 0},
    {256, 256, 16, 16, 16, 0},
    {512, 512,  8, 16, 64, 0},
};

काष्ठा ggml_opencl_fa_dim_table {
    स्थिर ggml_opencl_fa_dim * तथ्य;
    माप_प्रकार                     गिनती;

    स्थिर ggml_opencl_fa_dim * begin() स्थिर { वापस तथ्य; }
    स्थिर ggml_opencl_fa_dim * end()   स्थिर { वापस तथ्य + गिनती; }
};

// Mutable copy of the active table; GGML_OPENCL_FA_TUNE patches entries here
// at backend init without touching the const source table.
अटल ggml_opencl_fa_dim g_fa_dims_runtime[
    माप(g_fa_dims_adreno_default) / माप(g_fa_dims_adreno_default[0])];

अटल ggml_opencl_fa_dim_table g_opencl_fa_dims = {
    g_fa_dims_adreno_default,
    माप(g_fa_dims_adreno_default) / माप(g_fa_dims_adreno_default[0]),
};

// GGML_OPENCL_FA_TUNE=dk:dv:bm:bn:nsplit:thr[,…] — patches matching entries
// in the active table at backend init, before the first FA kernel compiles.
// Unmatched (dk,dv) pairs are warned and ignored.
अटल व्योम ggml_opencl_fa_apply_env_overrides() {
    स्थिर अक्षर * e = std::दो_पर्या("GGML_OPENCL_FA_TUNE");
    अगर (!e || !e[0]) {
        वापस;
    }

    std::string s = e;
    माप_प्रकार pos = 0;
    जबतक (pos < s.आकार()) {
        माप_प्रकार comma = s.find(',', pos);
        std::string entry = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        पूर्णांक dk, dv, bm, bn, nsplit, thr;
        अगर (std::माला_पूछो(entry.c_str(), "%d:%d:%d:%d:%d:%d", &dk, &dv, &bm, &bn, &nsplit, &thr) == 6) {
            bool patched = false;
            क्रम (माप_प्रकार i = 0; i < g_opencl_fa_dims.गिनती; ++i) {
                ggml_opencl_fa_dim & d = g_fa_dims_runtime[i];
                अगर (d.dk == dk && d.dv == dv) {
                    d.bm = bm; d.bn = bn; d.n_split = nsplit; d.nkv_split_threshold = thr;
                    GGML_LOG_INFO("ggml_opencl: FA tune override DK=%d DV=%d -> bm=%d bn=%d n_split=%d thr=%d\n",
                                  dk, dv, bm, bn, nsplit, thr);
                    patched = true;
                    अवरोध;
                }
            }
            अगर (!patched) {
                GGML_LOG_WARN("ggml_opencl: FA tune override DK=%d DV=%d ignored (no matching dim)\n", dk, dv);
            }
        } अन्यथा {
            GGML_LOG_WARN("ggml_opencl: FA tune override entry malformed: '%s'\n", entry.c_str());
        }
        अगर (comma == std::string::npos) अवरोध;
        pos = comma + 1;
    }
}

// Copy the default table into the mutable runtime buffer and apply any
// GGML_OPENCL_FA_TUNE overrides. A per-generation table can be added here
// once it has been tuned on hardware.
अटल व्योम ggml_cl_init_fa_dims_table() {
    स्थिर माप_प्रकार गिनती = माप(g_fa_dims_adreno_default) / माप(g_fa_dims_adreno_default[0]);
    क्रम (माप_प्रकार i = 0; i < गिनती; ++i) {
        g_fa_dims_runtime[i] = g_fa_dims_adreno_default[i];
    }
    g_opencl_fa_dims = { g_fa_dims_runtime, गिनती };
    ggml_opencl_fa_apply_env_overrides();
}
