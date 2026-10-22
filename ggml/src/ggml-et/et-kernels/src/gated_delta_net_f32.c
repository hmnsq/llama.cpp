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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_gated_delta_net_params {
    ರಚನೆ ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    ರಚನೆ ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    ರಚನೆ ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    ರಚನೆ ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    ರಚನೆ ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    ರಚನೆ ggml_tensor state_ಒಳ;  // [S_v*S_v*H, K, n_seqs]
    ರಚನೆ ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    int32_t            S_v;       // head dimension
    int32_t            H;         // number of value heads
    int32_t            H_q;       // number of Q heads
    int32_t            H_k;       // number of K heads
    int32_t            n_ಟೋಕನುಗಳು;  // total tokens
    int32_t            n_seqs;    // number of sequences
    int32_t            n_seqs_q;  // Q sequence count
    int32_t            n_seqs_k;  // K sequence count
    int32_t            kda;       // 1 if per-element gate, 0 if scalar
    int32_t            K;         // snapshot slot count
    ದಶಮಾಂಶ              scale;     // 1/sqrt(S_v)
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum_f10(ಶೂನ್ಯ) {
    ದಶಮಾಂಶ ಫಲಿತಾಂಶ;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(ಫಲಿತಾಂಶ)::"t0", "f1", "f2", "f3", "f4", "f5");
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_gated_delta_net_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * q_tsr     = &params->q;
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * k_tsr     = &params->k;
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * v_tsr     = &params->v;
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * g_tsr     = &params->g;
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * beta_tsr  = &params->beta;
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * state_tsr = &params->state_ಒಳ;
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst_tsr   = &params->dst;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q        = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) q_tsr->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k        = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) k_tsr->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v        = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) v_tsr->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g        = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) g_tsr->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * beta     = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) beta_tsr->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * state_ಒಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) state_tsr->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst_tsr->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int32_t S_v      = params->S_v;
    ಸ್ಥಿರಾಂಕ int32_t H        = params->H;
    ಸ್ಥಿರಾಂಕ int32_t H_q      = params->H_q;
    ಸ್ಥಿರಾಂಕ int32_t H_k      = params->H_k;
    ಸ್ಥಿರಾಂಕ int32_t n_ಟೋಕನುಗಳು = params->n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ int32_t n_seqs   = params->n_seqs;
    ಸ್ಥಿರಾಂಕ int32_t n_seqs_q = params->n_seqs_q;
    ಸ್ಥಿರಾಂಕ int32_t n_seqs_k = params->n_seqs_k;
    ಸ್ಥಿರಾಂಕ int32_t kda      = params->kda;
    ಸ್ಥಿರಾಂಕ int32_t K        = params->K;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   scale    = params->scale;

    ಆದರೆ (!q || !k || !v || !g || !beta || !state_ಒಳ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Preserve the original contract for every tensor except q, k, and v, which may be
    // row-contiguous with strided higher dimensions.
    ಆದರೆ (q_tsr->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || k_tsr->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || v_tsr->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) ||
        g_tsr->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || beta_tsr->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || state_tsr->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) ||
        dst_tsr->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int32_t attn_elems     = S_v * H * n_ಟೋಕನುಗಳು * n_seqs;
    ದಶಮಾಂಶ *       attn_ಹೊರಹೋಗು_ತಳ  = dst_ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       state_ಹೊರಹೋಗು_ತಳ = dst_ದತ್ತಾಂಶ + attn_elems;

    ಸ್ಥಿರಾಂಕ int32_t state_plane_floats  = S_v * S_v * H * n_seqs;

    ಸ್ಥಿರಾಂಕ int32_t G0 = kda ? S_v : 1;

    ಸ್ಥಿರಾಂಕ size_t  q_nb1      = q_tsr->nb[1];
    ಸ್ಥಿರಾಂಕ size_t  q_nb2      = q_tsr->nb[2];
    ಸ್ಥಿರಾಂಕ size_t  q_nb3      = q_tsr->nb[3];
    ಸ್ಥಿರಾಂಕ size_t  k_nb1      = k_tsr->nb[1];
    ಸ್ಥಿರಾಂಕ size_t  k_nb2      = k_tsr->nb[2];
    ಸ್ಥಿರಾಂಕ size_t  k_nb3      = k_tsr->nb[3];
    ಸ್ಥಿರಾಂಕ size_t  v_nb1      = v_tsr->nb[1];
    ಸ್ಥಿರಾಂಕ size_t  v_nb2      = v_tsr->nb[2];
    ಸ್ಥಿರಾಂಕ size_t  v_nb3      = v_tsr->nb[3];
    ಸ್ಥಿರಾಂಕ int32_t g_stride_h = G0;
    ಸ್ಥಿರಾಂಕ int32_t g_stride_t = G0 * H;
    ಸ್ಥಿರಾಂಕ int32_t g_stride_s = G0 * H * n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ int32_t b_stride_t = H;
    ಸ್ಥಿರಾಂಕ int32_t b_stride_s = H * n_ಟೋಕನುಗಳು;

    ದಶಮಾಂಶ exp_g_buf[128];

    // FP and SIMD share the same register file. Scalar FP needs the default
    // mask; 8-wide .ps blocks need m0=255. Save once, toggle at boundaries.
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ default_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %[ms]\n" : [ms] "=r"(default_ಮುಖವಾಡ));

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
    ಸ್ಥಿರಾಂಕ int32_t J_BLK      = ET_CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು / (int32_t) ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int32_t n_j_ಖಂಡಗಳು = (S_v + J_BLK - 1) / J_BLK;
    ಸ್ಥಿರಾಂಕ int32_t ಒಟ್ಟು_work = n_j_ಖಂಡಗಳು * H * n_seqs;

    ಗಾಗಿ (int32_t ir = thread_id; ir < ಒಟ್ಟು_work; ir += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int32_t jb   = ir % n_j_ಖಂಡಗಳು;
        ಸ್ಥಿರಾಂಕ int32_t ತಲೆ = (ir / n_j_ಖಂಡಗಳು) % H;
        ಸ್ಥಿರಾಂಕ int32_t seq  = ir / (n_j_ಖಂಡಗಳು * H);

        ಸ್ಥಿರಾಂಕ int32_t j_ಆರಂಭ = jb * J_BLK;
        ಸ್ಥಿರಾಂಕ int32_t j_ಕೊನೆ   = (j_ಆರಂಭ + J_BLK < S_v) ? j_ಆರಂಭ + J_BLK : S_v;

        ಸ್ಥಿರಾಂಕ int32_t h_q   = ತಲೆ % H_q;
        ಸ್ಥಿರಾಂಕ int32_t h_k   = ತಲೆ % H_k;
        ಸ್ಥಿರಾಂಕ int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        ಸ್ಥಿರಾಂಕ int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        ಸ್ಥಿರಾಂಕ int32_t ತಲೆ_state_off = (seq * H + ತಲೆ) * S_v * S_v;
        // Live RMW buffer = first snapshot plane (slot 0).
        ದಶಮಾಂಶ *       s_ಹೊರಹೋಗು          = state_ಹೊರಹೋಗು_ತಳ + ತಲೆ_state_off;
        // Input state: seq `seq`, head `head`.
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s_ಒಳ           = state_ಒಳ + ತಲೆ_state_off;

        // Skip the explicit s_in -> s_out copy. At t=0 pass A/B read through
        // src_state = s_in; pass B writes the first new row to s_out. From
        // t=1 onward src_state flips to s_out (read-modify-write in place).
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_state = s_ಒಳ;

        ಸ್ಥಿರಾಂಕ int32_t attn_stride_t = S_v * H;
        ದಶಮಾಂಶ *       attn_ptr      = attn_ಹೊರಹೋಗು_ತಳ + (seq * n_ಟೋಕನುಗಳು * H + ತಲೆ) * S_v;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ zero = 0.0f;

        ಗಾಗಿ (int32_t t = 0; t < n_ಟೋಕನುಗಳು; t++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_t      = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) q + seq_q * q_nb3 + t * q_nb2 + h_q * q_nb1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_t      = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k + seq_k * k_nb3 + t * k_nb2 + h_k * k_nb1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_t      = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v + seq * v_nb3 + t * v_nb2 + ತಲೆ * v_nb1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_t      = g + seq * g_stride_s + t * g_stride_t + ತಲೆ * g_stride_h;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   beta_val = beta[seq * b_stride_s + t * b_stride_t + ತಲೆ];

            // Precompute per-element gate for the kda path; scalar decay
            // otherwise. Decay is fused into per-j pass A/B below, not
            // applied to state in a separate pre-pass.
            ದಶಮಾಂಶ decay = 0.0f;  // only used when !kda
            ಆದರೆ (kda) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ log2e = 1.4426950408889634f;
                __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 255\n" :::);
                __asm__ ಅಸ್ಥಿರ("fbc.ps f20, %[l2e]\n" : : [l2e] "m"(log2e) : "f20");
                ಗಾಗಿ (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ ಅಸ್ಥಿರ(
                        "flw.ps f10, %[g_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fexp.ps f10, f10\n"
                        "fsw.ps f10, %[out]\n"
                        : [ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & exp_g_buf[i])
                        : [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & g_t[i])
                        : "f10");
                }
                __asm__ ಅಸ್ಥಿರ("mova.m.x %[ms]\n" : : [ms] "r"(default_ಮುಖವಾಡ));
            } ಇಲ್ಲದಿದ್ದರೆ {
                decay = et_expf(g_t[0]);
            }

            ಗಾಗಿ (int32_t j = j_ಆರಂಭ; j < j_ಕೊನೆ; j++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = src_state + j * S_v;
                ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು = s_ಹೊರಹೋಗು + j * S_v;

                __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 255\n" :::);
                ಆದರೆ (kda) {
                    __asm__ ಅಸ್ಥಿರ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
                    ಗಾಗಿ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ಅಡ್ಡಸಾಲು[i]),
                              [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & k_t[i])
                            : "f10", "f11", "f12", "f13");
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    __asm__ ಅಸ್ಥಿರ(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [d] "m"(decay)
                        : "f10", "f22");
                    ಗಾಗಿ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ಅಡ್ಡಸಾಲು[i]), [k_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) &
                                                                                           k_t[i])
                            : "f10", "f11", "f13");
                    }
                }

                ದಶಮಾಂಶ dot_sk = hsum_f10();
                __asm__ ಅಸ್ಥಿರ("mova.m.x %[ms]\n" : : [ms] "r"(default_ಮುಖವಾಡ));

                ದಶಮಾಂಶ delta_j = (v_t[j] - dot_sk) * beta_val;

                // -------- Pass B: decay + outer product + attn --------
                __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 255\n" :::);
                ಆದರೆ (kda) {
                    __asm__ ಅಸ್ಥಿರ(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j)
                        : "f10", "f21");
                    ಗಾಗಿ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[i])
                            : [s_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ಅಡ್ಡಸಾಲು[i]),
                              [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & k_t[i]), [q_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & q_t[i])
                            : "f10", "f11", "f12", "f13", "f14");
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    __asm__ ಅಸ್ಥಿರ(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j), [d] "m"(decay)
                        : "f10", "f21", "f22");
                    ಗಾಗಿ (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[i])
                            : [s_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ಅಡ್ಡಸಾಲು[i]),
                              [k_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & k_t[i]), [q_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & q_t[i])
                            : "f10", "f11", "f13", "f14");
                    }
                }

                ದಶಮಾಂಶ attn_val = hsum_f10();
                __asm__ ಅಸ್ಥಿರ("mova.m.x %[ms]\n" : : [ms] "r"(default_ಮುಖವಾಡ));

                attn_ptr[j] = attn_val * scale;
            }

            // n-way merge snapshot: live state lives in slot 0 (== s_out).
            // Copies state to target snapshot slots [1, K-1] in reverse chronological order.
            // target_slot == 0 is the live buffer itself => no copy.
            // target_slot >= K (when n_tokens > K) => older slots are discarded.
            ಆದರೆ (K > 1) {
                ಸ್ಥಿರಾಂಕ int32_t ಗುರಿ_ಸ್ಥಾನ = (n_ಟೋಕನುಗಳು - 1) - t;
                ಆದರೆ (ಗುರಿ_ಸ್ಥಾನ > 0 && ಗುರಿ_ಸ್ಥಾನ < K) {
                    ದಶಮಾಂಶ * snap = state_ಹೊರಹೋಗು_ತಳ + ಗುರಿ_ಸ್ಥಾನ * state_plane_floats + ತಲೆ_state_off;
                    ಗಾಗಿ (int32_t j = j_ಆರಂಭ; j < j_ಕೊನೆ; j++) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src = s_ಹೊರಹೋಗು + j * S_v;
                        ದಶಮಾಂಶ *       dst = snap + j * S_v;
                        ಗಾಗಿ (int32_t i = 0; i < S_v; i++) {
                            dst[i] = src[i];
                        }
                    }
                }
            }

            // After t=0, state lives in s_out; flip src_state so subsequent
            // timesteps read-modify-write in place.
            src_state = s_ಹೊರಹೋಗು;
            attn_ptr += attn_stride_t;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
