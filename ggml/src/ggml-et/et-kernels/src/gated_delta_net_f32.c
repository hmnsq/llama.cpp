//******************************************************************************
// Gated Delta Net F32 Kernel
//
// Implements the gated delta rule recurrence:
//   For each head h, timestep t:
//     1. Gate decay:   S *= exp(g)  (scalar or per-element KDA)
//     2. Delta update: delta[j] = (v[j] - dot(S_row_j, k)) * beta
//     3. Outer product: S_row_j += k * delta[j]
//     4. Attention:    attn[j] = dot(S_row_j, q) * scale
//
// State is stored transposed: s_out[j*S_v + i] = S[i][j]
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_gated_delta_net_params {
    ਬਣਤਰ ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    ਬਣਤਰ ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    ਬਣਤਰ ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    ਬਣਤਰ ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    ਬਣਤਰ ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    ਬਣਤਰ ggml_tensor state_ਵਿੱਚ;  // [S_v*S_v*H, K, n_seqs]
    ਬਣਤਰ ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    int32_t            S_v;       // head dimension
    int32_t            H;         // number of value heads
    int32_t            H_q;       // number of Q heads
    int32_t            H_k;       // number of K heads
    int32_t            n_ਟੋਕਨ;  // total tokens
    int32_t            n_seqs;    // number of sequences
    int32_t            n_seqs_q;  // Q sequence count
    int32_t            n_seqs_k;  // K sequence count
    int32_t            kda;       // 1 if per-element gate, 0 if scalar
    int32_t            K;         // snapshot slot count
    ਦਸ਼ਮਲਵ              scale;     // 1/sqrt(S_v)
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum_f10(ਖਾਲੀ) {
    ਦਸ਼ਮਲਵ ਨਤੀਜਾ;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(ਨਤੀਜਾ)::"t0", "f1", "f2", "f3", "f4", "f5");
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_gated_delta_net_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਬਣਤਰ ggml_tensor * q_tsr     = &params->q;
    ਅਚਲ ਬਣਤਰ ggml_tensor * k_tsr     = &params->k;
    ਅਚਲ ਬਣਤਰ ggml_tensor * v_tsr     = &params->v;
    ਅਚਲ ਬਣਤਰ ggml_tensor * g_tsr     = &params->g;
    ਅਚਲ ਬਣਤਰ ggml_tensor * beta_tsr  = &params->beta;
    ਅਚਲ ਬਣਤਰ ggml_tensor * state_tsr = &params->state_ਵਿੱਚ;
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst_tsr   = &params->dst;

    ਅਚਲ ਦਸ਼ਮਲਵ * q        = (ਅਚਲ ਦਸ਼ਮਲਵ *) q_tsr->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * k        = (ਅਚਲ ਦਸ਼ਮਲਵ *) k_tsr->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * v        = (ਅਚਲ ਦਸ਼ਮਲਵ *) v_tsr->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * g        = (ਅਚਲ ਦਸ਼ਮਲਵ *) g_tsr->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * beta     = (ਅਚਲ ਦਸ਼ਮਲਵ *) beta_tsr->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * state_ਵਿੱਚ = (ਅਚਲ ਦਸ਼ਮਲਵ *) state_tsr->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst_tsr->ਅੰਕੜੇ;

    ਅਚਲ int32_t S_v      = params->S_v;
    ਅਚਲ int32_t H        = params->H;
    ਅਚਲ int32_t H_q      = params->H_q;
    ਅਚਲ int32_t H_k      = params->H_k;
    ਅਚਲ int32_t n_ਟੋਕਨ = params->n_ਟੋਕਨ;
    ਅਚਲ int32_t n_seqs   = params->n_seqs;
    ਅਚਲ int32_t n_seqs_q = params->n_seqs_q;
    ਅਚਲ int32_t n_seqs_k = params->n_seqs_k;
    ਅਚਲ int32_t kda      = params->kda;
    ਅਚਲ int32_t K        = params->K;
    ਅਚਲ ਦਸ਼ਮਲਵ   scale    = params->scale;

    ਜੇ (!q || !k || !v || !g || !beta || !state_ਵਿੱਚ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    // Preserve the original contract for every tensor except q, k, and v, which may be
    // row-contiguous with strided higher dimensions.
    ਜੇ (q_tsr->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || k_tsr->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || v_tsr->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) ||
        g_tsr->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || beta_tsr->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || state_tsr->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) ||
        dst_tsr->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int32_t attn_elems     = S_v * H * n_ਟੋਕਨ * n_seqs;
    ਦਸ਼ਮਲਵ *       attn_ਬਾਹਰਲਾ_ਅਧਾਰ  = dst_ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       state_ਬਾਹਰਲਾ_ਅਧਾਰ = dst_ਅੰਕੜੇ + attn_elems;

    ਅਚਲ int32_t state_plane_floats  = S_v * S_v * H * n_seqs;

    ਅਚਲ int32_t G0 = kda ? S_v : 1;

    ਅਚਲ size_t  q_nb1      = q_tsr->nb[1];
    ਅਚਲ size_t  q_nb2      = q_tsr->nb[2];
    ਅਚਲ size_t  q_nb3      = q_tsr->nb[3];
    ਅਚਲ size_t  k_nb1      = k_tsr->nb[1];
    ਅਚਲ size_t  k_nb2      = k_tsr->nb[2];
    ਅਚਲ size_t  k_nb3      = k_tsr->nb[3];
    ਅਚਲ size_t  v_nb1      = v_tsr->nb[1];
    ਅਚਲ size_t  v_nb2      = v_tsr->nb[2];
    ਅਚਲ size_t  v_nb3      = v_tsr->nb[3];
    ਅਚਲ int32_t g_stride_h = G0;
    ਅਚਲ int32_t g_stride_t = G0 * H;
    ਅਚਲ int32_t g_stride_s = G0 * H * n_ਟੋਕਨ;
    ਅਚਲ int32_t b_stride_t = H;
    ਅਚਲ int32_t b_stride_s = H * n_ਟੋਕਨ;

    ਦਸ਼ਮਲਵ exp_g_buf[128];

    // FP and SIMD share the same register file. Scalar FP needs the default
    // mask; 8-wide .ps blocks need m0=255. Save once, toggle at boundaries.
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ default_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %[ms]\n" : [ms] "=r"(default_ਨਕਾਬ));

    // Parallelize over (j_block, head, seq).  J_BLK must satisfy two separate
    // cache-line alignment constraints at once:
    //   (a) State: J_BLK consecutive rows of s_out (each S_v floats) span an
    //       integer number of cache lines.  For S_v * sizeof(float) >= 64 this
    //       is trivially any J_BLK >= 1.
    //   (b) Attention output: each j writes exactly one float into
    //       attn_ptr[j], which is densely packed.  If J_BLK * sizeof(float) is
    //       less than a cache line, distinct threads will share a line and
    //       race on scalar stores — ET's L1 isn't coherent so we lose writes.
    //
    // (b) dominates: J_BLK must be at least ET_CACHE_LINE_SIZE_BYTES / 4 so
    // that each thread owns a whole cache line of attn_ptr.  That's 16 on
    // ET-SoC-1, and it's also a whole number of state rows for every
    // S_v >= 1, so (a) is automatically satisfied.
    ਅਚਲ int32_t J_BLK      = ET_CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ / (int32_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int32_t n_j_ਖੰਡਾਂ = (S_v + J_BLK - 1) / J_BLK;
    ਅਚਲ int32_t ਕੁੱਲ_work = n_j_ਖੰਡਾਂ * H * n_seqs;

    ਲਈ (int32_t ir = thread_id; ir < ਕੁੱਲ_work; ir += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int32_t jb   = ir % n_j_ਖੰਡਾਂ;
        ਅਚਲ int32_t ਸਿਰ = (ir / n_j_ਖੰਡਾਂ) % H;
        ਅਚਲ int32_t seq  = ir / (n_j_ਖੰਡਾਂ * H);

        ਅਚਲ int32_t j_ਸ਼ੁਰੂ = jb * J_BLK;
        ਅਚਲ int32_t j_ਸਿਰਾ   = (j_ਸ਼ੁਰੂ + J_BLK < S_v) ? j_ਸ਼ੁਰੂ + J_BLK : S_v;

        ਅਚਲ int32_t h_q   = ਸਿਰ % H_q;
        ਅਚਲ int32_t h_k   = ਸਿਰ % H_k;
        ਅਚਲ int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        ਅਚਲ int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        ਅਚਲ int32_t ਸਿਰ_state_off = (seq * H + ਸਿਰ) * S_v * S_v;
        // Live RMW buffer = first snapshot plane (slot 0).
        ਦਸ਼ਮਲਵ *       s_ਬਾਹਰਲਾ          = state_ਬਾਹਰਲਾ_ਅਧਾਰ + ਸਿਰ_state_off;
        // Input state: seq `seq`, head `head`.
        ਅਚਲ ਦਸ਼ਮਲਵ * s_ਵਿੱਚ           = state_ਵਿੱਚ + ਸਿਰ_state_off;

        // Skip the explicit s_in -> s_out copy. At t=0 pass A/B read through
        // src_state = s_in; pass B writes the first new row to s_out. From
        // t=1 onward src_state flips to s_out (read-modify-write in place).
        ਅਚਲ ਦਸ਼ਮਲਵ * src_state = s_ਵਿੱਚ;

        ਅਚਲ int32_t attn_stride_t = S_v * H;
        ਦਸ਼ਮਲਵ *       attn_ptr      = attn_ਬਾਹਰਲਾ_ਅਧਾਰ + (seq * n_ਟੋਕਨ * H + ਸਿਰ) * S_v;

        ਅਚਲ ਦਸ਼ਮਲਵ zero = 0.0f;

        ਲਈ (int32_t t = 0; t < n_ਟੋਕਨ; t++) {
            ਅਚਲ ਦਸ਼ਮਲਵ * q_t      = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) q + seq_q * q_nb3 + t * q_nb2 + h_q * q_nb1);
            ਅਚਲ ਦਸ਼ਮਲਵ * k_t      = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) k + seq_k * k_nb3 + t * k_nb2 + h_k * k_nb1);
            ਅਚਲ ਦਸ਼ਮਲਵ * v_t      = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) v + seq * v_nb3 + t * v_nb2 + ਸਿਰ * v_nb1);
            ਅਚਲ ਦਸ਼ਮਲਵ * g_t      = g + seq * g_stride_s + t * g_stride_t + ਸਿਰ * g_stride_h;
            ਅਚਲ ਦਸ਼ਮਲਵ   beta_val = beta[seq * b_stride_s + t * b_stride_t + ਸਿਰ];

            // Precompute per-element gate for the kda path; scalar decay
            // otherwise. Decay is fused into per-j pass A/B below, not
            // applied to state in a separate pre-pass.
            ਦਸ਼ਮਲਵ decay = 0.0f;  // only used when !kda
            ਜੇ (kda) {
                ਅਚਲ ਦਸ਼ਮਲਵ log2e = 1.4426950408889634f;
                __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 255\n" :::);
                __asm__ ਅਸਥਿਰ("fbc.ps f20, %[l2e]\n" : : [l2e] "m"(log2e) : "f20");
                ਲਈ (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ ਅਸਥਿਰ(
                        "flw.ps f10, %[g_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fexp.ps f10, f10\n"
                        "fsw.ps f10, %[out]\n"
                        : [ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & exp_g_buf[i])
                        : [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & g_t[i])
                        : "f10");
                }
                __asm__ ਅਸਥਿਰ("mova.m.x %[ms]\n" : : [ms] "r"(default_ਨਕਾਬ));
            } ਨਹੀਂਤਾਂ {
                decay = et_expf(g_t[0]);
            }

            ਲਈ (int32_t j = j_ਸ਼ੁਰੂ; j < j_ਸਿਰਾ; j++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = src_state + j * S_v;
                ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ = s_ਬਾਹਰਲਾ + j * S_v;

                __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 255\n" :::);
                ਜੇ (kda) {
                    __asm__ ਅਸਥਿਰ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
                    ਲਈ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ਅਸਥਿਰ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ਪੰਕਤੀ[i]),
                              [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & k_t[i])
                            : "f10", "f11", "f12", "f13");
                    }
                } ਨਹੀਂਤਾਂ {
                    __asm__ ਅਸਥਿਰ(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [d] "m"(decay)
                        : "f10", "f22");
                    ਲਈ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ਅਸਥਿਰ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ਪੰਕਤੀ[i]), [k_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) &
                                                                                           k_t[i])
                            : "f10", "f11", "f13");
                    }
                }

                ਦਸ਼ਮਲਵ dot_sk = hsum_f10();
                __asm__ ਅਸਥਿਰ("mova.m.x %[ms]\n" : : [ms] "r"(default_ਨਕਾਬ));

                ਦਸ਼ਮਲਵ delta_j = (v_t[j] - dot_sk) * beta_val;

                // -------- Pass B: decay + outer product + attn --------
                __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 255\n" :::);
                ਜੇ (kda) {
                    __asm__ ਅਸਥਿਰ(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j)
                        : "f10", "f21");
                    ਲਈ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ਅਸਥਿਰ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[i])
                            : [s_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ਪੰਕਤੀ[i]),
                              [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & k_t[i]), [q_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & q_t[i])
                            : "f10", "f11", "f12", "f13", "f14");
                    }
                } ਨਹੀਂਤਾਂ {
                    __asm__ ਅਸਥਿਰ(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j), [d] "m"(decay)
                        : "f10", "f21", "f22");
                    ਲਈ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ਅਸਥਿਰ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[i])
                            : [s_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ਪੰਕਤੀ[i]),
                              [k_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & k_t[i]), [q_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & q_t[i])
                            : "f10", "f11", "f13", "f14");
                    }
                }

                ਦਸ਼ਮਲਵ attn_val = hsum_f10();
                __asm__ ਅਸਥਿਰ("mova.m.x %[ms]\n" : : [ms] "r"(default_ਨਕਾਬ));

                attn_ptr[j] = attn_val * scale;
            }

            // n-way merge snapshot: live state lives in slot 0 (== s_out).
            // Copies state to target snapshot slots [1, K-1] in reverse chronological order.
            // target_slot == 0 is the live buffer itself => no copy.
            // target_slot >= K (when n_tokens > K) => older slots are discarded.
            ਜੇ (K > 1) {
                ਅਚਲ int32_t ਨਿਸ਼ਾਨਾ_ਥਾਂ = (n_ਟੋਕਨ - 1) - t;
                ਜੇ (ਨਿਸ਼ਾਨਾ_ਥਾਂ > 0 && ਨਿਸ਼ਾਨਾ_ਥਾਂ < K) {
                    ਦਸ਼ਮਲਵ * snap = state_ਬਾਹਰਲਾ_ਅਧਾਰ + ਨਿਸ਼ਾਨਾ_ਥਾਂ * state_plane_floats + ਸਿਰ_state_off;
                    ਲਈ (int32_t j = j_ਸ਼ੁਰੂ; j < j_ਸਿਰਾ; j++) {
                        ਅਚਲ ਦਸ਼ਮਲਵ * src = s_ਬਾਹਰਲਾ + j * S_v;
                        ਦਸ਼ਮਲਵ *       dst = snap + j * S_v;
                        ਲਈ (int32_t i = 0; i < S_v; i++) {
                            dst[i] = src[i];
                        }
                    }
                }
            }

            // After t=0, state lives in s_out; flip src_state so subsequent
            // timesteps read-modify-write in place.
            src_state = s_ਬਾਹਰਲਾ;
            attn_ptr += attn_stride_t;
        }
    }

    ਵਾਪਸ 0;
}
