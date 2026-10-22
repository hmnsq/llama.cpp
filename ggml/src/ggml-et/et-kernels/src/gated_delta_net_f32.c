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

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_gated_delta_net_params {
    بنية ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    بنية ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    بنية ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    بنية ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    بنية ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    بنية ggml_tensor state_درون;  // [S_v*S_v*H, K, n_seqs]
    بنية ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    int32_t            S_v;       // head dimension
    int32_t            H;         // number of value heads
    int32_t            H_q;       // number of Q heads
    int32_t            H_k;       // number of K heads
    int32_t            n_نشانهها;  // total tokens
    int32_t            n_seqs;    // number of sequences
    int32_t            n_seqs_q;  // Q sequence count
    int32_t            n_seqs_k;  // K sequence count
    int32_t            kda;       // 1 if per-element gate, 0 if scalar
    int32_t            K;         // snapshot slot count
    عائم              scale;     // 1/sqrt(S_v)
};

ساكن مضمن عائم hsum_f10(فراغ) {
    عائم نتیجه;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(نتیجه)::"t0", "f1", "f2", "f3", "f4", "f5");
    أرجع نتیجه;
}

صحيح مدخل_نقطه(بنية ggml_et_gated_delta_net_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    ثابت بنية ggml_tensor * q_tsr     = &params->q;
    ثابت بنية ggml_tensor * k_tsr     = &params->k;
    ثابت بنية ggml_tensor * v_tsr     = &params->v;
    ثابت بنية ggml_tensor * g_tsr     = &params->g;
    ثابت بنية ggml_tensor * beta_tsr  = &params->beta;
    ثابت بنية ggml_tensor * state_tsr = &params->state_درون;
    ثابت بنية ggml_tensor * dst_tsr   = &params->dst;

    ثابت عائم * q        = (ثابت عائم *) q_tsr->داده;
    ثابت عائم * k        = (ثابت عائم *) k_tsr->داده;
    ثابت عائم * v        = (ثابت عائم *) v_tsr->داده;
    ثابت عائم * g        = (ثابت عائم *) g_tsr->داده;
    ثابت عائم * beta     = (ثابت عائم *) beta_tsr->داده;
    ثابت عائم * state_درون = (ثابت عائم *) state_tsr->داده;
    عائم *       dst_داده = (عائم *) dst_tsr->داده;

    ثابت int32_t S_v      = params->S_v;
    ثابت int32_t H        = params->H;
    ثابت int32_t H_q      = params->H_q;
    ثابت int32_t H_k      = params->H_k;
    ثابت int32_t n_نشانهها = params->n_نشانهها;
    ثابت int32_t n_seqs   = params->n_seqs;
    ثابت int32_t n_seqs_q = params->n_seqs_q;
    ثابت int32_t n_seqs_k = params->n_seqs_k;
    ثابت int32_t kda      = params->kda;
    ثابت int32_t K        = params->K;
    ثابت عائم   scale    = params->scale;

    إذا (!q || !k || !v || !g || !beta || !state_درون || !dst_داده) {
        أرجع -1;
    }

    // Preserve the original contract for every tensor except q, k, and v, which may be
    // row-contiguous with strided higher dimensions.
    إذا (q_tsr->nb[0] != حجم(عائم) || k_tsr->nb[0] != حجم(عائم) || v_tsr->nb[0] != حجم(عائم) ||
        g_tsr->nb[0] != حجم(عائم) || beta_tsr->nb[0] != حجم(عائم) || state_tsr->nb[0] != حجم(عائم) ||
        dst_tsr->nb[0] != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int32_t attn_elems     = S_v * H * n_نشانهها * n_seqs;
    عائم *       attn_بیرون_پایه  = dst_داده;
    عائم *       state_بیرون_پایه = dst_داده + attn_elems;

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
    ثابت int32_t g_stride_s = G0 * H * n_نشانهها;
    ثابت int32_t b_stride_t = H;
    ثابت int32_t b_stride_s = H * n_نشانهها;

    عائم exp_g_buf[128];

    // FP and SIMD share the same register file. Scalar FP needs the default
    // mask; 8-wide .ps blocks need m0=255. Save once, toggle at boundaries.
    بیعلامت دراز default_ماسک;
    __asm__ متطاير("mova.x.m %[ms]\n" : [ms] "=r"(default_ماسک));

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
    ثابت int32_t J_BLK      = ET_CACHE_سطر_اندازه_بایتها / (int32_t) حجم(عائم);
    ثابت int32_t n_j_بلوکها = (S_v + J_BLK - 1) / J_BLK;
    ثابت int32_t مجموع_work = n_j_بلوکها * H * n_seqs;

    لكل (int32_t ir = thread_id; ir < مجموع_work; ir += رقم_نخها) {
        ثابت int32_t jb   = ir % n_j_بلوکها;
        ثابت int32_t سر = (ir / n_j_بلوکها) % H;
        ثابت int32_t seq  = ir / (n_j_بلوکها * H);

        ثابت int32_t j_شروع = jb * J_BLK;
        ثابت int32_t j_انتها   = (j_شروع + J_BLK < S_v) ? j_شروع + J_BLK : S_v;

        ثابت int32_t h_q   = سر % H_q;
        ثابت int32_t h_k   = سر % H_k;
        ثابت int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        ثابت int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        ثابت int32_t سر_state_off = (seq * H + سر) * S_v * S_v;
        // Live RMW buffer = first snapshot plane (slot 0).
        عائم *       s_بیرون          = state_بیرون_پایه + سر_state_off;
        // Input state: seq `seq`, head `head`.
        ثابت عائم * s_درون           = state_درون + سر_state_off;

        // Skip the explicit s_in -> s_out copy. At t=0 pass A/B read through
        // src_state = s_in; pass B writes the first new row to s_out. From
        // t=1 onward src_state flips to s_out (read-modify-write in place).
        ثابت عائم * src_state = s_درون;

        ثابت int32_t attn_stride_t = S_v * H;
        عائم *       attn_ptr      = attn_بیرون_پایه + (seq * n_نشانهها * H + سر) * S_v;

        ثابت عائم zero = 0.0f;

        لكل (int32_t t = 0; t < n_نشانهها; t++) {
            ثابت عائم * q_t      = (ثابت عائم *) ((ثابت حرف *) q + seq_q * q_nb3 + t * q_nb2 + h_q * q_nb1);
            ثابت عائم * k_t      = (ثابت عائم *) ((ثابت حرف *) k + seq_k * k_nb3 + t * k_nb2 + h_k * k_nb1);
            ثابت عائم * v_t      = (ثابت عائم *) ((ثابت حرف *) v + seq * v_nb3 + t * v_nb2 + سر * v_nb1);
            ثابت عائم * g_t      = g + seq * g_stride_s + t * g_stride_t + سر * g_stride_h;
            ثابت عائم   beta_val = beta[seq * b_stride_s + t * b_stride_t + سر];

            // Precompute per-element gate for the kda path; scalar decay
            // otherwise. Decay is fused into per-j pass A/B below, not
            // applied to state in a separate pre-pass.
            عائم decay = 0.0f;  // only used when !kda
            إذا (kda) {
                ثابت عائم log2e = 1.4426950408889634f;
                __asm__ متطاير("mov.m.x m0, x0, 255\n" :::);
                __asm__ متطاير("fbc.ps f20, %[l2e]\n" : : [l2e] "m"(log2e) : "f20");
                لكل (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ متطاير(
                        "flw.ps f10, %[g_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fexp.ps f10, f10\n"
                        "fsw.ps f10, %[out]\n"
                        : [بیرون] "=m"(*(عائم (*)[8]) & exp_g_buf[i])
                        : [g_vec] "m"(*(ثابت عائم (*)[8]) & g_t[i])
                        : "f10");
                }
                __asm__ متطاير("mova.m.x %[ms]\n" : : [ms] "r"(default_ماسک));
            } وإلا {
                decay = et_expf(g_t[0]);
            }

            لكل (int32_t j = j_شروع; j < j_انتها; j++) {
                ثابت عائم * src_ردیف = src_state + j * S_v;
                عائم *       dst_ردیف = s_بیرون + j * S_v;

                __asm__ متطاير("mov.m.x m0, x0, 255\n" :::);
                إذا (kda) {
                    __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
                    لكل (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متطاير(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ثابت عائم (*)[8]) & src_ردیف[i]),
                              [g_vec] "m"(*(ثابت عائم (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ثابت عائم (*)[8]) & k_t[i])
                            : "f10", "f11", "f12", "f13");
                    }
                } وإلا {
                    __asm__ متطاير(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [d] "m"(decay)
                        : "f10", "f22");
                    لكل (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متطاير(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(ثابت عائم (*)[8]) & src_ردیف[i]), [k_vec] "m"(*(ثابت عائم (*)[8]) &
                                                                                           k_t[i])
                            : "f10", "f11", "f13");
                    }
                }

                عائم dot_sk = hsum_f10();
                __asm__ متطاير("mova.m.x %[ms]\n" : : [ms] "r"(default_ماسک));

                عائم delta_j = (v_t[j] - dot_sk) * beta_val;

                // -------- Pass B: decay + outer product + attn --------
                __asm__ متطاير("mov.m.x m0, x0, 255\n" :::);
                إذا (kda) {
                    __asm__ متطاير(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j)
                        : "f10", "f21");
                    لكل (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متطاير(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_بیرون] "=m"(*(عائم (*)[8]) & dst_ردیف[i])
                            : [s_vec] "m"(*(ثابت عائم (*)[8]) & src_ردیف[i]),
                              [g_vec] "m"(*(ثابت عائم (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(ثابت عائم (*)[8]) & k_t[i]), [q_vec] "m"(*(ثابت عائم (*)[8]) & q_t[i])
                            : "f10", "f11", "f12", "f13", "f14");
                    }
                } وإلا {
                    __asm__ متطاير(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j), [d] "m"(decay)
                        : "f10", "f21", "f22");
                    لكل (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ متطاير(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_بیرون] "=m"(*(عائم (*)[8]) & dst_ردیف[i])
                            : [s_vec] "m"(*(ثابت عائم (*)[8]) & src_ردیف[i]),
                              [k_vec] "m"(*(ثابت عائم (*)[8]) & k_t[i]), [q_vec] "m"(*(ثابت عائم (*)[8]) & q_t[i])
                            : "f10", "f11", "f13", "f14");
                    }
                }

                عائم attn_val = hsum_f10();
                __asm__ متطاير("mova.m.x %[ms]\n" : : [ms] "r"(default_ماسک));

                attn_ptr[j] = attn_val * scale;
            }

            // n-way merge snapshot: live state lives in slot 0 (== s_out).
            // Copies state to target snapshot slots [1, K-1] in reverse chronological order.
            // target_slot == 0 is the live buffer itself => no copy.
            // target_slot >= K (when n_tokens > K) => older slots are discarded.
            إذا (K > 1) {
                ثابت int32_t هدف_جایگاه = (n_نشانهها - 1) - t;
                إذا (هدف_جایگاه > 0 && هدف_جایگاه < K) {
                    عائم * snap = state_بیرون_پایه + هدف_جایگاه * state_plane_floats + سر_state_off;
                    لكل (int32_t j = j_شروع; j < j_انتها; j++) {
                        ثابت عائم * src = s_بیرون + j * S_v;
                        عائم *       dst = snap + j * S_v;
                        لكل (int32_t i = 0; i < S_v; i++) {
                            dst[i] = src[i];
                        }
                    }
                }
            }

            // After t=0, state lives in s_out; flip src_state so subsequent
            // timesteps read-modify-write in place.
            src_state = s_بیرون;
            attn_ptr += attn_stride_t;
        }
    }

    أرجع 0;
}
