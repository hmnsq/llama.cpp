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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_gated_delta_net_params {
    నిర్మాణం ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    నిర్మాణం ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    నిర్మాణం ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    నిర్మాణం ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    నిర్మాణం ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    నిర్మాణం ggml_tensor state_లోపల;  // [S_v*S_v*H, K, n_seqs]
    నిర్మాణం ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    int32_t            S_v;       // head dimension
    int32_t            H;         // number of value heads
    int32_t            H_q;       // number of Q heads
    int32_t            H_k;       // number of K heads
    int32_t            n_టోకెన్లు;  // total tokens
    int32_t            n_seqs;    // number of sequences
    int32_t            n_seqs_q;  // Q sequence count
    int32_t            n_seqs_k;  // K sequence count
    int32_t            kda;       // 1 if per-element gate, 0 if scalar
    int32_t            K;         // snapshot slot count
    దశాంశం              scale;     // 1/sqrt(S_v)
};

స్థిర అంతఃవరుస దశాంశం hsum_f10(శూన్య) {
    దశాంశం ఫలితం;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(ఫలితం)::"t0", "f1", "f2", "f3", "f4", "f5");
    తిరిగి ఫలితం;
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_gated_delta_net_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి నిర్మాణం ggml_tensor * q_tsr     = &params->q;
    స్థిరరాశి నిర్మాణం ggml_tensor * k_tsr     = &params->k;
    స్థిరరాశి నిర్మాణం ggml_tensor * v_tsr     = &params->v;
    స్థిరరాశి నిర్మాణం ggml_tensor * g_tsr     = &params->g;
    స్థిరరాశి నిర్మాణం ggml_tensor * beta_tsr  = &params->beta;
    స్థిరరాశి నిర్మాణం ggml_tensor * state_tsr = &params->state_లోపల;
    స్థిరరాశి నిర్మాణం ggml_tensor * dst_tsr   = &params->dst;

    స్థిరరాశి దశాంశం * q        = (స్థిరరాశి దశాంశం *) q_tsr->దత్తాంశం;
    స్థిరరాశి దశాంశం * k        = (స్థిరరాశి దశాంశం *) k_tsr->దత్తాంశం;
    స్థిరరాశి దశాంశం * v        = (స్థిరరాశి దశాంశం *) v_tsr->దత్తాంశం;
    స్థిరరాశి దశాంశం * g        = (స్థిరరాశి దశాంశం *) g_tsr->దత్తాంశం;
    స్థిరరాశి దశాంశం * beta     = (స్థిరరాశి దశాంశం *) beta_tsr->దత్తాంశం;
    స్థిరరాశి దశాంశం * state_లోపల = (స్థిరరాశి దశాంశం *) state_tsr->దత్తాంశం;
    దశాంశం *       dst_దత్తాంశం = (దశాంశం *) dst_tsr->దత్తాంశం;

    స్థిరరాశి int32_t S_v      = params->S_v;
    స్థిరరాశి int32_t H        = params->H;
    స్థిరరాశి int32_t H_q      = params->H_q;
    స్థిరరాశి int32_t H_k      = params->H_k;
    స్థిరరాశి int32_t n_టోకెన్లు = params->n_టోకెన్లు;
    స్థిరరాశి int32_t n_seqs   = params->n_seqs;
    స్థిరరాశి int32_t n_seqs_q = params->n_seqs_q;
    స్థిరరాశి int32_t n_seqs_k = params->n_seqs_k;
    స్థిరరాశి int32_t kda      = params->kda;
    స్థిరరాశి int32_t K        = params->K;
    స్థిరరాశి దశాంశం   scale    = params->scale;

    ఒకవేళ (!q || !k || !v || !g || !beta || !state_లోపల || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    // Preserve the original contract for every tensor except q, k, and v, which may be
    // row-contiguous with strided higher dimensions.
    ఒకవేళ (q_tsr->nb[0] != పరిమాణం(దశాంశం) || k_tsr->nb[0] != పరిమాణం(దశాంశం) || v_tsr->nb[0] != పరిమాణం(దశాంశం) ||
        g_tsr->nb[0] != పరిమాణం(దశాంశం) || beta_tsr->nb[0] != పరిమాణం(దశాంశం) || state_tsr->nb[0] != పరిమాణం(దశాంశం) ||
        dst_tsr->nb[0] != పరిమాణం(దశాంశం)) {
        తిరిగి -1;
    }

    స్థిరరాశి int32_t attn_elems     = S_v * H * n_టోకెన్లు * n_seqs;
    దశాంశం *       attn_బహిర్గతం_పునాది  = dst_దత్తాంశం;
    దశాంశం *       state_బహిర్గతం_పునాది = dst_దత్తాంశం + attn_elems;

    స్థిరరాశి int32_t state_plane_floats  = S_v * S_v * H * n_seqs;

    స్థిరరాశి int32_t G0 = kda ? S_v : 1;

    స్థిరరాశి size_t  q_nb1      = q_tsr->nb[1];
    స్థిరరాశి size_t  q_nb2      = q_tsr->nb[2];
    స్థిరరాశి size_t  q_nb3      = q_tsr->nb[3];
    స్థిరరాశి size_t  k_nb1      = k_tsr->nb[1];
    స్థిరరాశి size_t  k_nb2      = k_tsr->nb[2];
    స్థిరరాశి size_t  k_nb3      = k_tsr->nb[3];
    స్థిరరాశి size_t  v_nb1      = v_tsr->nb[1];
    స్థిరరాశి size_t  v_nb2      = v_tsr->nb[2];
    స్థిరరాశి size_t  v_nb3      = v_tsr->nb[3];
    స్థిరరాశి int32_t g_stride_h = G0;
    స్థిరరాశి int32_t g_stride_t = G0 * H;
    స్థిరరాశి int32_t g_stride_s = G0 * H * n_టోకెన్లు;
    స్థిరరాశి int32_t b_stride_t = H;
    స్థిరరాశి int32_t b_stride_s = H * n_టోకెన్లు;

    దశాంశం exp_g_buf[128];

    // FP and SIMD share the same register file. Scalar FP needs the default
    // mask; 8-wide .ps blocks need m0=255. Save once, toggle at boundaries.
    గుర్తులేని దీర్ఘ default_ముసుగు;
    __asm__ అస్థిర("mova.x.m %[ms]\n" : [ms] "=r"(default_ముసుగు));

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
    స్థిరరాశి int32_t J_BLK      = ET_CACHE_పంక్తి_కొలత_బైట్లు / (int32_t) పరిమాణం(దశాంశం);
    స్థిరరాశి int32_t n_j_ఖండాలు = (S_v + J_BLK - 1) / J_BLK;
    స్థిరరాశి int32_t మొత్తం_work = n_j_ఖండాలు * H * n_seqs;

    కొరకు (int32_t ir = thread_id; ir < మొత్తం_work; ir += అంకె_దారాలు) {
        స్థిరరాశి int32_t jb   = ir % n_j_ఖండాలు;
        స్థిరరాశి int32_t తల = (ir / n_j_ఖండాలు) % H;
        స్థిరరాశి int32_t seq  = ir / (n_j_ఖండాలు * H);

        స్థిరరాశి int32_t j_ప్రారంభం = jb * J_BLK;
        స్థిరరాశి int32_t j_చివర   = (j_ప్రారంభం + J_BLK < S_v) ? j_ప్రారంభం + J_BLK : S_v;

        స్థిరరాశి int32_t h_q   = తల % H_q;
        స్థిరరాశి int32_t h_k   = తల % H_k;
        స్థిరరాశి int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        స్థిరరాశి int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        స్థిరరాశి int32_t తల_state_off = (seq * H + తల) * S_v * S_v;
        // Live RMW buffer = first snapshot plane (slot 0).
        దశాంశం *       s_బహిర్గతం          = state_బహిర్గతం_పునాది + తల_state_off;
        // Input state: seq `seq`, head `head`.
        స్థిరరాశి దశాంశం * s_లోపల           = state_లోపల + తల_state_off;

        // Skip the explicit s_in -> s_out copy. At t=0 pass A/B read through
        // src_state = s_in; pass B writes the first new row to s_out. From
        // t=1 onward src_state flips to s_out (read-modify-write in place).
        స్థిరరాశి దశాంశం * src_state = s_లోపల;

        స్థిరరాశి int32_t attn_stride_t = S_v * H;
        దశాంశం *       attn_ptr      = attn_బహిర్గతం_పునాది + (seq * n_టోకెన్లు * H + తల) * S_v;

        స్థిరరాశి దశాంశం zero = 0.0f;

        కొరకు (int32_t t = 0; t < n_టోకెన్లు; t++) {
            స్థిరరాశి దశాంశం * q_t      = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) q + seq_q * q_nb3 + t * q_nb2 + h_q * q_nb1);
            స్థిరరాశి దశాంశం * k_t      = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) k + seq_k * k_nb3 + t * k_nb2 + h_k * k_nb1);
            స్థిరరాశి దశాంశం * v_t      = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) v + seq * v_nb3 + t * v_nb2 + తల * v_nb1);
            స్థిరరాశి దశాంశం * g_t      = g + seq * g_stride_s + t * g_stride_t + తల * g_stride_h;
            స్థిరరాశి దశాంశం   beta_val = beta[seq * b_stride_s + t * b_stride_t + తల];

            // Precompute per-element gate for the kda path; scalar decay
            // otherwise. Decay is fused into per-j pass A/B below, not
            // applied to state in a separate pre-pass.
            దశాంశం decay = 0.0f;  // only used when !kda
            ఒకవేళ (kda) {
                స్థిరరాశి దశాంశం log2e = 1.4426950408889634f;
                __asm__ అస్థిర("mov.m.x m0, x0, 255\n" :::);
                __asm__ అస్థిర("fbc.ps f20, %[l2e]\n" : : [l2e] "m"(log2e) : "f20");
                కొరకు (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ అస్థిర(
                        "flw.ps f10, %[g_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fexp.ps f10, f10\n"
                        "fsw.ps f10, %[out]\n"
                        : [బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & exp_g_buf[i])
                        : [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & g_t[i])
                        : "f10");
                }
                __asm__ అస్థిర("mova.m.x %[ms]\n" : : [ms] "r"(default_ముసుగు));
            } లేకపోతే {
                decay = et_expf(g_t[0]);
            }

            కొరకు (int32_t j = j_ప్రారంభం; j < j_చివర; j++) {
                స్థిరరాశి దశాంశం * src_అడ్డువరుస = src_state + j * S_v;
                దశాంశం *       dst_అడ్డువరుస = s_బహిర్గతం + j * S_v;

                __asm__ అస్థిర("mov.m.x m0, x0, 255\n" :::);
                ఒకవేళ (kda) {
                    __asm__ అస్థిర("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
                    కొరకు (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ అస్థిర(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_అడ్డువరుస[i]),
                              [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & k_t[i])
                            : "f10", "f11", "f12", "f13");
                    }
                } లేకపోతే {
                    __asm__ అస్థిర(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [d] "m"(decay)
                        : "f10", "f22");
                    కొరకు (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ అస్థిర(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_అడ్డువరుస[i]), [k_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) &
                                                                                           k_t[i])
                            : "f10", "f11", "f13");
                    }
                }

                దశాంశం dot_sk = hsum_f10();
                __asm__ అస్థిర("mova.m.x %[ms]\n" : : [ms] "r"(default_ముసుగు));

                దశాంశం delta_j = (v_t[j] - dot_sk) * beta_val;

                // -------- Pass B: decay + outer product + attn --------
                __asm__ అస్థిర("mov.m.x m0, x0, 255\n" :::);
                ఒకవేళ (kda) {
                    __asm__ అస్థిర(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j)
                        : "f10", "f21");
                    కొరకు (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ అస్థిర(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[i])
                            : [s_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_అడ్డువరుస[i]),
                              [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & k_t[i]), [q_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & q_t[i])
                            : "f10", "f11", "f12", "f13", "f14");
                    }
                } లేకపోతే {
                    __asm__ అస్థిర(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j), [d] "m"(decay)
                        : "f10", "f21", "f22");
                    కొరకు (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ అస్థిర(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[i])
                            : [s_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_అడ్డువరుస[i]),
                              [k_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & k_t[i]), [q_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & q_t[i])
                            : "f10", "f11", "f13", "f14");
                    }
                }

                దశాంశం attn_val = hsum_f10();
                __asm__ అస్థిర("mova.m.x %[ms]\n" : : [ms] "r"(default_ముసుగు));

                attn_ptr[j] = attn_val * scale;
            }

            // n-way merge snapshot: live state lives in slot 0 (== s_out).
            // Copies state to target snapshot slots [1, K-1] in reverse chronological order.
            // target_slot == 0 is the live buffer itself => no copy.
            // target_slot >= K (when n_tokens > K) => older slots are discarded.
            ఒకవేళ (K > 1) {
                స్థిరరాశి int32_t లక్ష్యం_స్థానం = (n_టోకెన్లు - 1) - t;
                ఒకవేళ (లక్ష్యం_స్థానం > 0 && లక్ష్యం_స్థానం < K) {
                    దశాంశం * snap = state_బహిర్గతం_పునాది + లక్ష్యం_స్థానం * state_plane_floats + తల_state_off;
                    కొరకు (int32_t j = j_ప్రారంభం; j < j_చివర; j++) {
                        స్థిరరాశి దశాంశం * src = s_బహిర్గతం + j * S_v;
                        దశాంశం *       dst = snap + j * S_v;
                        కొరకు (int32_t i = 0; i < S_v; i++) {
                            dst[i] = src[i];
                        }
                    }
                }
            }

            // After t=0, state lives in s_out; flip src_state so subsequent
            // timesteps read-modify-write in place.
            src_state = s_బహిర్గతం;
            attn_ptr += attn_stride_t;
        }
    }

    తిరిగి 0;
}
