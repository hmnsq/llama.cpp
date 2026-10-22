#సూచన once

#అన్తర్భావ "ggml-et-common.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <inttypes.h>

// Performance logging macros for ET ops
// Logs in machine-parseable pipe-delimited format: ET_PERF|field=value|...
#నిర్వచితమైతే ET_PERF_రికార్డు
#    define ET_PERF_ప్రారంభం() int64_t et_perf_ప్రారంభం = ggml_సమయం_us()

#    define ET_PERF_చివర(op_name, kernel_name, కణుపు)                                                                \
        చేయి {                                                                                                       \
            int64_t et_perf_చివర      = ggml_సమయం_us();                                                            \
            int64_t _et_perf_duration = et_perf_చివర - et_perf_ప్రారంభం;                                             \
            GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 \
                           ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "\n",                  \
                           op_name, kernel_name, _et_perf_duration, (కణుపు)->name, (కణుపు)->ne[0], (కణుపు)->ne[1],    \
                           (కణుపు)->ne[2], (కణుపు)->ne[3], et_perf_ప్రారంభం, et_perf_చివర);                            \
        } అయితే (0)

#    define ET_PERF_చివర_EXT(op_name, kernel_name, కణుపు, fmt, ...)                                                  \
        చేయి {                                                                                                       \
            int64_t et_perf_చివర      = ggml_సమయం_us();                                                            \
            int64_t _et_perf_duration = et_perf_చివర - et_perf_ప్రారంభం;                                             \
            GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 \
                           ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "|" fmt "\n",          \
                           op_name, kernel_name, _et_perf_duration, (కణుపు)->name, (కణుపు)->ne[0], (కణుపు)->ne[1],    \
                           (కణుపు)->ne[2], (కణుపు)->ne[3], et_perf_ప్రారంభం, et_perf_చివర, ##__VA_ARGS__);             \
        } అయితే (0)
#లేకపోతే

#    define ET_PERF_ప్రారంభం() \
        చేయి {                \
        } అయితే (0)
#    define ET_PERF_చివర_EXT(op_name, kernel_name, కణుపు, fmt, ...) \
        చేయి {                                                      \
            (శూన్య) (కణుపు);                                        \
        } అయితే (0)
#    define ET_PERF_చివర(op_name, kernel_name, కణుపు) \
        చేయి {                                        \
            (శూన్య) (కణుపు);                          \
        } అయితే (0)

#అయితే_ముగింపు  // ET_PERF_RECORD

నిర్మాణం ggml_et_ద్విమాన_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

// Q8_0 mul_mat with optional residual bias.
// bias.data == NULL means "no bias" - kernel skips the add.
// When non-NULL, bias must have the same shape and strides as dst.
నిర్మాణం ggml_et_mm_q8_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    ggml_tensor bias;
};

నిర్మాణం ggml_et_im2col_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

// Element map parameters for embarrassingly parallel binary operations (MUL, ADD, etc.)
// Operation type is determined by dst->op (GGML_OP_MUL, GGML_OP_ADD, etc.)
నిర్మాణం ggml_et_elmap_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

నిర్మాణం ggml_et_rope_settings {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t బహుళకం;    // ROPE mode, GGML_ROPE_TYPE_*
    int32_t n_ctx;
    int32_t n_ctx_orig;
    దశాంశం   freq_పునాది;    // Base frequency (usually 10000.0f)
    దశాంశం   freq_scale;   // Frequency scaling factor
    దశాంశం   ext_గుణకం;   // Extension factor for YaRN
    దశాంశం   attn_గుణకం;  // Attention factor for YaRN
    దశాంశం   beta_fast;    // Fast beta for YaRN
    దశాంశం   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
};

నిర్మాణం ggml_et_rope_params {
    ggml_tensor           src0;
    ggml_tensor           src1;
    ggml_tensor           src2;
    ggml_tensor           dst;
    ggml_et_rope_settings rope_params;
};

నిర్మాణం ggml_et_rms_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    దశాంశం       eps;   // Epsilon parameter for numerical stability
};

నిర్మాణం ggml_et_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    దశాంశం       eps;   // Epsilon parameter for numerical stability
};

నిర్మాణం ggml_et_l2_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    దశాంశం       eps;   // Epsilon parameter for numerical stability
};

నిర్మాణం ggml_et_సమూహం_norm_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor dst;       // F32 output tensor
    int32_t     n_groups;  // Number of channel groups
    దశాంశం       eps;       // Epsilon parameter for numerical stability
};

నిర్మాణం ggml_et_glu_params {
    ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t     glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t     swapped;      // Whether gate and value are swapped
    దశాంశం       alpha;        // SWIGLU_OAI: sigmoid scaling factor (unused for other variants)
    దశాంశం       హద్దు;        // SWIGLU_OAI: clamp limit (unused for other variants)
};

నిర్మాణం ggml_et_softmax_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    ggml_tensor dst;       // F32 output tensor
    దశాంశం       scale;     // Scale factor
    దశాంశం       max_bias;  // Max bias for ALiBi (0.0f if not used)
};

నిర్మాణం ggml_et_flash_attn_ext_params {
    ggml_tensor src0;      // Q tensor (F32)
    ggml_tensor src1;      // K tensor (F32)
    ggml_tensor src2;      // V tensor (F32)
    ggml_tensor ముసుగు;      // mask tensor (F16 or F32), zeroed when absent
    ggml_tensor dst;       // Output tensor (F32)
    దశాంశం       scale;     // Scale factor applied to QK
    int32_t     has_ముసుగు;  // nonzero if mask is present
};

నిర్మాణం ggml_et_get_అడ్డువరుసలు_params {
    ggml_tensor src0;  // Data tensor (F32 or Q8_0)
    ggml_tensor src1;  // Row indices tensor (I32)
    ggml_tensor dst;   // Output tensor (F32)
};

నిర్మాణం ggml_et_cont_params {
    ggml_tensor src0;  // F32 input tensor (non-contiguous)
    ggml_tensor dst;   // F32 output tensor (contiguous)
};

నిర్మాణం ggml_et_concat_params {
    ggml_tensor src0;  // F32 input tensor 0
    ggml_tensor src1;  // F32 input tensor 1
    ggml_tensor dst;   // F32 output tensor
    int32_t     dim;   // Concatenation dimension
};

నిర్మాణం ggml_et_repeat_params {
    ggml_tensor src0;  // F32 input tensor (tile)
    ggml_tensor dst;   // F32 output tensor (tiled result)
};

నిర్మాణం ggml_et_fill_params {
    ggml_tensor dst;  // F32 output tensor (contiguous)
    దశాంశం       c;    // Constant value to fill
};

నిర్మాణం ggml_et_tri_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor dst;       // F32 output tensor
    int32_t     tri_type;  // ggml_tri_type enum value
};

నిర్మాణం ggml_et_solve_tri_params {
    ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

నిర్మాణం ggml_et_pad_params {
    ggml_tensor src0;   // F32 input (may be non-contiguous, nb[0] must == 4)
    ggml_tensor dst;    // F32 output (contiguous, ne[0] % 16 == 0)
    int32_t     lp[4];  // left padding per dimension
    int32_t     rp[4];  // right padding per dimension
};

నిర్మాణం ggml_et_diag_params {
    ggml_tensor src0;  // F32 input vector
    ggml_tensor dst;   // F32 output diagonal matrix
};

నిర్మాణం ggml_et_ssm_conv_params {
    ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

నిర్మాణం ggml_et_ssm_పరిశీలన_params {
    ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    ggml_tensor src6;  // ids: [n_seqs] i32
    ggml_tensor dst;   // [y, states] packed output from ggml_ssm_scan()
    int32_t     K;
};

నిర్మాణం ggml_et_rwkv_wkv6_params {
    దశాంశం * k;         // src[0]: [S, H, T]  key
    దశాంశం * v;         // src[1]: [S, H, T]  value
    దశాంశం * r;         // src[2]: [S, H, T]  receptance
    దశాంశం * tf;        // src[3]: [S, H]     time_faaaa (per-head)
    దశాంశం * td;        // src[4]: [S, H, T]  time_decay
    దశాంశం * state_లోపల;  // src[5]: [S*S*H, n_seqs]  initial state
    దశాంశం * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

నిర్మాణం ggml_et_rwkv_wkv7_params {
    దశాంశం * r;         // [S, H, T]  receptance
    దశాంశం * w;         // [S, H, T]  decay
    దశాంశం * k;         // [S, H, T]  key
    దశాంశం * v;         // [S, H, T]  value
    దశాంశం * a;         // [S, H, T]  bonus gate
    దశాంశం * b;         // [S, H, T]  bonus key
    దశాంశం * state_లోపల;  // [S*S*H, n_seqs]  initial state
    దశాంశం * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

నిర్మాణం ggml_et_gated_delta_net_params {
    ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    ggml_tensor state_లోపల;  // [S_v*S_v*H, K, n_seqs]
    ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    int32_t     S_v;       // head dimension (value size)
    int32_t     H;         // number of value heads
    int32_t     H_q;       // number of Q heads
    int32_t     H_k;       // number of K heads
    int32_t     n_టోకెన్లు;  // total tokens
    int32_t     n_seqs;    // number of sequences (from V)
    int32_t     n_seqs_q;  // Q sequence count
    int32_t     n_seqs_k;  // K sequence count
    int32_t     kda;       // 1 if per-element gate (g_ne0 == S_v), 0 if scalar
    int32_t     K;         // snapshot slot count
    దశాంశం       scale;     // 1/sqrt(S_v)
};

నిర్మాణం ggml_et_సమితి_అడ్డువరుసలు_params {
    ggml_tensor src0;  // F32 source data tensor
    ggml_tensor src1;  // I64 row indices tensor
    ggml_tensor dst;   // F32/F16 destination tensor
};

నిర్మాణం ggml_et_సమితి_params {
    ggml_tensor src1;    // F32 source view to write into dst
    ggml_tensor dst;     // F32 destination/base tensor
    int32_t     nb1;     // destination view stride for dim 1
    int32_t     nb2;     // destination view stride for dim 2
    int32_t     nb3;     // destination view stride for dim 3
    int32_t     స్థానభ్రంశం;  // byte offset into destination
};

నిర్మాణం ggml_et_rms_norm_mul_params {
    ggml_tensor src0;  // F32 input tensor (to be normalized)
    ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    ggml_tensor dst;   // F32 output tensor
    దశాంశం       eps;   // Epsilon for numerical stability
};

నిర్మాణం ggml_et_mul_mat_id_params {
    ggml_tensor src0;  // Expert weight matrices (Q8_0/F16/F32) [K, M, n_expert]
    ggml_tensor src1;  // Activations (F32) [K, n_expert_used, batch]
    ggml_tensor src2;  // Expert indices (I32) [n_expert_used, batch]
    ggml_tensor dst;   // Output (F32) [M, n_expert_used, batch, 1]
};

నిర్మాణం ggml_et_sqr_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
};

నిర్మాణం ggml_et_unary_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor dst;       // F32 output tensor
    int32_t     unary_op;  // ggml_unary_op enum value
};

నిర్మాణం ggml_et_కూడిక_అడ్డువరుసలు_params {
    ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

నిర్మాణం ggml_et_సగటు_params {
    ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

నిర్మాణం ggml_et_clamp_params {
    ggml_tensor src0;  // F32 input tensor (contiguous)
    ggml_tensor dst;   // F32 output tensor (contiguous; may alias src0)
    దశాంశం       min_val;
    దశాంశం       max_val;
};

నిర్మాణం ggml_et_cumsum_params {
    ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ggml_tensor dst;   // F32 output tensor [ne00, ne01, ne02, ne03]
};

నిర్మాణం ggml_et_scale_params {
    ggml_tensor src0;   // F32 input tensor
    ggml_tensor dst;    // F32 output tensor
    దశాంశం       scale;  // Scale factor
    దశాంశం       bias;   // Bias (additive offset)
};

bool ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_కూడిక_అడ్డువరుసలు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_సగటు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_కూడు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
// add_node is optional: when non-NULL and the pair (node, add_node) was
// validated by ggml_et_can_fuse({MUL_MAT, ADD}), the Q8_0 path writes
// dst = mm(...) + add_node's "other" operand (the bias) in one launch.
bool ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        స్థిరరాశి ggml_tensor *              కణుపు,
                        స్థిరరాశి ggml_tensor *              కూడు_కణుపు = nullptr);
bool ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_సమూహం_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_get_అడ్డువరుసలు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_సమితి_అడ్డువరుసలు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_సమితి(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_ssm_పరిశీలన(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు);
bool ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             స్థిరరాశి ggml_tensor *              rms_norm_కణుపు,
                             స్థిరరాశి ggml_tensor *              mul_కణుపు);
