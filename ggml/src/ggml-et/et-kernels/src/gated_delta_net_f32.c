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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_gated_delta_net_params {
    ساخت ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    ساخت ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    ساخت ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    ساخت ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    ساخت ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    ساخت ggml_tensor state_اندر;  // [S_v*S_v*H, K, n_seqs]
    ساخت ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    int32_t            S_v;       // head dimension
    int32_t            H;         // number of value heads
    int32_t            H_q;       // number of Q heads
    int32_t            H_k;       // number of K heads
    int32_t            n_علامات;  // total tokens
    int32_t            n_seqs;    // number of sequences
    int32_t            n_seqs_q;  // Q sequence count
    int32_t            n_seqs_k;  // K sequence count
    int32_t            kda;       // 1 if per-element gate, 0 if scalar
    int32_t            K;         // snapshot slot count
    اعشاری              scale;     // 1/sqrt(S_v)
};

ساکن اندرونی اعشاری hsum_f10(خالی) {
    اعشاری نتیجہ;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(نتیجہ)::"t0", "f1", "f2", "f3", "f4", "f5");
    واپس نتیجہ;
}

صحیح اندراج_نقطہ(ساخت ggml_et_gated_delta_net_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ثابت ساخت ggml_tensor * q_tsr     = &params->q;
    ثابت ساخت ggml_tensor * k_tsr     = &params->k;
    ثابت ساخت ggml_tensor * v_tsr     = &params->v;
    ثابت ساخت ggml_tensor * g_tsr     = &params->g;
    ثابت ساخت ggml_tensor * beta_tsr  = &params->beta;
    ثابت ساخت ggml_tensor * state_tsr = &params->state_اندر;
    ثابت ساخت ggml_tensor * dst_tsr   = &params->dst;

    ثابت اعشاری * q        = (ثابت اعشاری *) q_tsr->ڈیٹا;
    ثابت اعشاری * k        = (ثابت اعشاری *) k_tsr->ڈیٹا;
    ثابت اعشاری * v        = (ثابت اعشاری *) v_tsr->ڈیٹا;
    ثابت اعشاری * g        = (ثابت اعشاری *) g_tsr->ڈیٹا;
    ثابت اعشاری * beta     = (ثابت اعشاری *) beta_tsr->ڈیٹا;
    ثابت اعشاری * state_اندر = (ثابت اعشاری *) state_tsr->ڈیٹا;
    اعشاری *       dst_ڈیٹا = (اعشاری *) dst_tsr->ڈیٹا;

    ثابت int32_t S_v      = params->S_v;
    ثابت int32_t H        = params->H;
    ثابت int32_t H_q      = params->H_q;
    ثابت int32_t H_k      = params->H_k;
    ثابت int32_t n_علامات = params->n_علامات;
    ثابت int32_t n_seqs   = params->n_seqs;
    ثابت int32_t n_seqs_q = params->n_seqs_q;
    ثابت int32_t n_seqs_k = params->n_seqs_k;
    ثابت int32_t kda      = params->kda;
    ثابت int32_t K        = params->K;
    ثابت اعشاری   scale    = params->scale;

    اگر (!q || !k || !v || !g || !beta || !state_اندر || !dst_ڈیٹا) {
        واپس -1;
    }

    // Preserve the original contract for every tensor except q, k, and v, which may be
    // row-contiguous with strided higher dimensions.
    اگر (q_tsr->nb[0] != حجم(اعشاری) || k_tsr->nb[0] != حجم(اعشاری) || v_tsr->nb[0] != حجم(اعشاری) ||
        g_tsr->nb[0] != حجم(اعشاری) || beta_tsr->nb[0] != حجم(اعشاری) || state_tsr->nb[0] != حجم(اعشاری) ||
        dst_tsr->nb[0] != حجم(اعشاری)) {
        واپس -1;
    }

    ثابت int32_t attn_elems     = S_v * H * n_علامات * n_seqs;
    اعشاری *       attn_برآمد_بنیاد  = dst_ڈیٹا;
    اعشاری *       state_برآمد_بنیاد = dst_ڈیٹا + attn_elems;

    ثابت int32_t state_plane_floats  = S_v * S_v * H * n_seqs;

    ثابت int32_t G0 = kda ? S_v : 1;

    ثابت size_t  q_nb1      = q_tsr->nb[1];
    ثابت size_t  q_nb2      = q_tsr->nb[2];
    ثابت size_t  q_nb3      = q_tsr->nb[3];
    ثابت size_t  k_nb1      = k_tsr->nb[1];
    ثابت size_t  k_nb2      = k_tsr->nb[2];
    ثابت size_t  k_nb3      = k_tsr->nb[3];
    ثابت size_t  v_nb1      = v_tsr->nb[1];
    ثابت size_t  v_nb2      = v_tsr->nb[2];
    ثابت size_t  v_nb3      = v_tsr->nb[3];
    ثابت int32_t g_stride_h = G0;
    ثابت int32_t g_stride_t = G0 * H;
    ثابت int32_t g_stride_s = G0 * H * n_علامات;
    ثابت int32_t b_stride_t = H;
    ثابت int32_t b_stride_s = H * n_علامات;

    اعشاری exp_g_buf[128];

    // FP and SIMD share the same register file. Scalar FP needs the default
    // mask; 8-wide .ps blocks need m0=255. Save once, toggle at boundaries.
    غیرعلامتی طویل default_نقاب;
    __asm__ متلون("mova.x.m %[ms]\n" : [ms] "=r"(default_نقاب));

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
    ثابت int32_t J_BLK      = ET_CACHE_سطر_سائز_بائٹیں / (int32_t) حجم(اعشاری);
    ثابت int32_t n_j_بلاکیں = (S_v + J_BLK - 1) / J_BLK;
    ثابت int32_t کل_work = n_j_بلاکیں * H * n_seqs;

    برائے (int32_t ir = thread_id; ir < کل_work; ir += عدد_دھاگے) {
        ثابت int32_t jb   = ir % n_j_بلاکیں;
        ثابت int32_t سر = (ir / n_j_بلاکیں) % H;
        ثابت int32_t seq  = ir / (n_j_بلاکیں * H);

        ثابت int32_t j_آغاز = jb * J_BLK;
        ثابت int32_t j_انتہا   = (j_آغاز + J_BLK < S_v) ? j_آغاز + J_BLK : S_v;

        ثابت int32_t h_q   = سر % H_q;
        ثابت int32_t h_k   = سر % H_k;
        ثابت int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        ثابت int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        ثابت int32_t سر_state_off = (seq * H + سر) * S_v * S_v;
        // Live RMW buffer = first snapshot plane (slot 0).
        اعشاری *       s_برآمد          = state_برآمد_بنیاد + سر_state_off;
        // Input state: seq `seq`, head `head`.
        ثابت اعشاری * s_اندر           = state_اندر + سر_state_off;

        // Skip the explicit s_in -> s_out copy. At t=0 pass A/B read through
        // src_state = s_in; pass B writes the first new row to s_out. From
        // t=1 onward src_state flips to s_out (read-modify-write in place).
        ثابت اعشاری * src_state = s_اندر;

        ثابت int32_t attn_stride_t = S_v * H;
        اعشاری *       attn_ptr      = attn_برآمد_بنیاد + (seq * n_علامات * H + سر) * S_v;

        ثابت اعشاری zero = 0.0f;

        برائے (int32_t t = 0; t < n_علامات; t++) {
            ثابت اعشاری * q_t      = (ثابت اعشاری *) ((ثابت حرف *) q + seq_q * q_nb3 + t * q_nb2 + h_q * q_nb1);
            ثابت اعشاری * k_t      = (ثابت اعشاری *) ((ثابت حرف *) k + seq_k * k_nb3 + t * k_nb2 + h_k * k_nb1);
            ثابت اعشاری * v_t      = (ثابت اعشاری *) ((ثابت حرف *) v + seq * v_nb3 + t * v_nb2 + سر * v_nb1);
            ثابت اعشاری * g_t      = g + seq * g_stride_s + t * g_stride_t + سر * g_stride_h;
            ثابت اعشاری   beta_val = beta[seq * b_stride_s + t * b_stride_t + سر];

            // Precompute per-element gate for the kda path; scalar decay
            // otherwise. Decay is fused into per-j pass A/B below, not
            // applied to state in a separate pre-pass.
            اعشاری decay = 0.0f;  // only used when !kda
            اگر (kda) {
                ثابت اعشاری log2e = 1.4426950408889634f;
                __asm__ متلون("mov.m.x m0, x0, 255\n" :::);
                __asm__ متلون("fbc.ps f20, %[l2e]\n" : : [l2e] "m"(log2e) : "f20");
                برائے (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ متلون(
                        "flw.ps f10, %[g_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fexp.ps f10, f10\n"
                        "fsw.ps f10, %[out]\n"
                        : [برآمد] "=m"(*(اعشاری (*)[8]) & exp_g_buf[i])
                        : [g_vec] "m"(*(ثابت اعشاری (*)[8]) & g_t[i])
                        : "f10");
                }
                __asm__ متلون("mova.m.x %[ms]\n" : : [ms] "r"(default_نقاب));
            } ورنہ {
                decay = et_expf(g_t[0]);
            }

            برائے (int32_t j = j_آغاز; j < j_انتہا; j++) {
                ثابت اعشاری * src_صف = src_state + j * S_v;
                اعشاری *       dst_صف = s_برآمد + j * S_v;

                __asm__ متلون("mov.m.x m0, x0, 255\n" :::);
                اگر (kda) {
                    __asm__ متلون("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
                    برائے (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ثابت اعشاری (*)[8]) & src_صف[i]),
                              [g_vec] "m"(*(ثابت اعشاری (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ثابت اعشاری (*)[8]) & k_t[i])
                            : "f10", "f11", "f12", "f13");
                    }
                } ورنہ {
                    __asm__ متلون(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [d] "m"(decay)
                        : "f10", "f22");
                    برائے (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ثابت اعشاری (*)[8]) & src_صف[i]), [k_vec] "m"(*(ثابت اعشاری (*)[8]) &
                                                                                           k_t[i])
                            : "f10", "f11", "f13");
                    }
                }

                اعشاری dot_sk = hsum_f10();
                __asm__ متلون("mova.m.x %[ms]\n" : : [ms] "r"(default_نقاب));

                اعشاری delta_j = (v_t[j] - dot_sk) * beta_val;

                // -------- Pass B: decay + outer product + attn --------
                __asm__ متلون("mov.m.x m0, x0, 255\n" :::);
                اگر (kda) {
                    __asm__ متلون(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j)
                        : "f10", "f21");
                    برائے (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_برآمد] "=m"(*(اعشاری (*)[8]) & dst_صف[i])
                            : [s_vec] "m"(*(ثابت اعشاری (*)[8]) & src_صف[i]),
                              [g_vec] "m"(*(ثابت اعشاری (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ثابت اعشاری (*)[8]) & k_t[i]), [q_vec] "m"(*(ثابت اعشاری (*)[8]) & q_t[i])
                            : "f10", "f11", "f12", "f13", "f14");
                    }
                } ورنہ {
                    __asm__ متلون(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j), [d] "m"(decay)
                        : "f10", "f21", "f22");
                    برائے (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متلون(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_برآمد] "=m"(*(اعشاری (*)[8]) & dst_صف[i])
                            : [s_vec] "m"(*(ثابت اعشاری (*)[8]) & src_صف[i]),
                              [k_vec] "m"(*(ثابت اعشاری (*)[8]) & k_t[i]), [q_vec] "m"(*(ثابت اعشاری (*)[8]) & q_t[i])
                            : "f10", "f11", "f13", "f14");
                    }
                }

                اعشاری attn_val = hsum_f10();
                __asm__ متلون("mova.m.x %[ms]\n" : : [ms] "r"(default_نقاب));

                attn_ptr[j] = attn_val * scale;
            }

            // n-way merge snapshot: live state lives in slot 0 (== s_out).
            // Copies state to target snapshot slots [1, K-1] in reverse chronological order.
            // target_slot == 0 is the live buffer itself => no copy.
            // target_slot >= K (when n_tokens > K) => older slots are discarded.
            اگر (K > 1) {
                ثابت int32_t ہدف_خانہ = (n_علامات - 1) - t;
                اگر (ہدف_خانہ > 0 && ہدف_خانہ < K) {
                    اعشاری * snap = state_برآمد_بنیاد + ہدف_خانہ * state_plane_floats + سر_state_off;
                    برائے (int32_t j = j_آغاز; j < j_انتہا; j++) {
                        ثابت اعشاری * src = s_برآمد + j * S_v;
                        اعشاری *       dst = snap + j * S_v;
                        برائے (int32_t i = 0; i < S_v; i++) {
                            dst[i] = src[i];
                        }
                    }
                }
            }

            // After t=0, state lives in s_out; flip src_state so subsequent
            // timesteps read-modify-write in place.
            src_state = s_برآمد;
            attn_ptr += attn_stride_t;
        }
    }

    واپس 0;
}
