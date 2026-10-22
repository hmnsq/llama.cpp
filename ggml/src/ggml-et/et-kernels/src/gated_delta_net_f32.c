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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_gated_delta_net_params {
    संरचना ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    संरचना ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    संरचना ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    संरचना ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    संरचना ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    संरचना ggml_tensor state_in;  // [S_v*S_v*H, K, n_seqs]
    संरचना ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    int32_t            S_v;       // head dimension
    int32_t            H;         // number of value heads
    int32_t            H_q;       // number of Q heads
    int32_t            H_k;       // number of K heads
    int32_t            n_tokens;  // total tokens
    int32_t            n_seqs;    // number of sequences
    int32_t            n_seqs_q;  // Q sequence count
    int32_t            n_seqs_k;  // K sequence count
    int32_t            kda;       // 1 if per-element gate, 0 if scalar
    int32_t            K;         // snapshot slot count
    दशांश              scale;     // 1/sqrt(S_v)
};

स्थिर ओळीत दशांश hsum_f10(शून्य) {
    दशांश result;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    परत result;
}

पूर्णांक entry_point(संरचना ggml_et_gated_delta_net_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    स्थिरांक संरचना ggml_tensor * q_tsr     = &params->q;
    स्थिरांक संरचना ggml_tensor * k_tsr     = &params->k;
    स्थिरांक संरचना ggml_tensor * v_tsr     = &params->v;
    स्थिरांक संरचना ggml_tensor * g_tsr     = &params->g;
    स्थिरांक संरचना ggml_tensor * beta_tsr  = &params->beta;
    स्थिरांक संरचना ggml_tensor * state_tsr = &params->state_in;
    स्थिरांक संरचना ggml_tensor * dst_tsr   = &params->dst;

    स्थिरांक दशांश * q        = (स्थिरांक दशांश *) q_tsr->data;
    स्थिरांक दशांश * k        = (स्थिरांक दशांश *) k_tsr->data;
    स्थिरांक दशांश * v        = (स्थिरांक दशांश *) v_tsr->data;
    स्थिरांक दशांश * g        = (स्थिरांक दशांश *) g_tsr->data;
    स्थिरांक दशांश * beta     = (स्थिरांक दशांश *) beta_tsr->data;
    स्थिरांक दशांश * state_in = (स्थिरांक दशांश *) state_tsr->data;
    दशांश *       dst_data = (दशांश *) dst_tsr->data;

    स्थिरांक int32_t S_v      = params->S_v;
    स्थिरांक int32_t H        = params->H;
    स्थिरांक int32_t H_q      = params->H_q;
    स्थिरांक int32_t H_k      = params->H_k;
    स्थिरांक int32_t n_tokens = params->n_tokens;
    स्थिरांक int32_t n_seqs   = params->n_seqs;
    स्थिरांक int32_t n_seqs_q = params->n_seqs_q;
    स्थिरांक int32_t n_seqs_k = params->n_seqs_k;
    स्थिरांक int32_t kda      = params->kda;
    स्थिरांक int32_t K        = params->K;
    स्थिरांक दशांश   scale    = params->scale;

    जर (!q || !k || !v || !g || !beta || !state_in || !dst_data) {
        परत -1;
    }

    // Preserve the original contract for every tensor except q, k, and v, which may be
    // row-contiguous with strided higher dimensions.
    जर (q_tsr->nb[0] != आकार(दशांश) || k_tsr->nb[0] != आकार(दशांश) || v_tsr->nb[0] != आकार(दशांश) ||
        g_tsr->nb[0] != आकार(दशांश) || beta_tsr->nb[0] != आकार(दशांश) || state_tsr->nb[0] != आकार(दशांश) ||
        dst_tsr->nb[0] != आकार(दशांश)) {
        परत -1;
    }

    स्थिरांक int32_t attn_elems     = S_v * H * n_tokens * n_seqs;
    दशांश *       attn_out_base  = dst_data;
    दशांश *       state_out_base = dst_data + attn_elems;

    स्थिरांक int32_t state_plane_floats  = S_v * S_v * H * n_seqs;

    स्थिरांक int32_t G0 = kda ? S_v : 1;

    स्थिरांक माप_प्रकार  q_nb1      = q_tsr->nb[1];
    स्थिरांक माप_प्रकार  q_nb2      = q_tsr->nb[2];
    स्थिरांक माप_प्रकार  q_nb3      = q_tsr->nb[3];
    स्थिरांक माप_प्रकार  k_nb1      = k_tsr->nb[1];
    स्थिरांक माप_प्रकार  k_nb2      = k_tsr->nb[2];
    स्थिरांक माप_प्रकार  k_nb3      = k_tsr->nb[3];
    स्थिरांक माप_प्रकार  v_nb1      = v_tsr->nb[1];
    स्थिरांक माप_प्रकार  v_nb2      = v_tsr->nb[2];
    स्थिरांक माप_प्रकार  v_nb3      = v_tsr->nb[3];
    स्थिरांक int32_t g_stride_h = G0;
    स्थिरांक int32_t g_stride_t = G0 * H;
    स्थिरांक int32_t g_stride_s = G0 * H * n_tokens;
    स्थिरांक int32_t b_stride_t = H;
    स्थिरांक int32_t b_stride_s = H * n_tokens;

    दशांश exp_g_buf[128];

    // FP and SIMD share the same register file. Scalar FP needs the default
    // mask; 8-wide .ps blocks need m0=255. Save once, toggle at boundaries.
    अचिन्हित दीर्घ default_mask;
    __asm__ अस्थिर("mova.x.m %[ms]\n" : [ms] "=r"(default_mask));

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
    स्थिरांक int32_t J_BLK      = ET_CACHE_ओळ_SIZE_BYTES / (int32_t) आकार(दशांश);
    स्थिरांक int32_t n_j_blocks = (S_v + J_BLK - 1) / J_BLK;
    स्थिरांक int32_t total_work = n_j_blocks * H * n_seqs;

    साठी (int32_t ir = thread_id; ir < total_work; ir += अंक_threads) {
        स्थिरांक int32_t jb   = ir % n_j_blocks;
        स्थिरांक int32_t head = (ir / n_j_blocks) % H;
        स्थिरांक int32_t seq  = ir / (n_j_blocks * H);

        स्थिरांक int32_t j_start = jb * J_BLK;
        स्थिरांक int32_t j_end   = (j_start + J_BLK < S_v) ? j_start + J_BLK : S_v;

        स्थिरांक int32_t h_q   = head % H_q;
        स्थिरांक int32_t h_k   = head % H_k;
        स्थिरांक int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        स्थिरांक int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        स्थिरांक int32_t head_state_off = (seq * H + head) * S_v * S_v;
        // Live RMW buffer = first snapshot plane (slot 0).
        दशांश *       s_out          = state_out_base + head_state_off;
        // Input state: seq `seq`, head `head`.
        स्थिरांक दशांश * s_in           = state_in + head_state_off;

        // Skip the explicit s_in -> s_out copy. At t=0 pass A/B read through
        // src_state = s_in; pass B writes the first new row to s_out. From
        // t=1 onward src_state flips to s_out (read-modify-write in place).
        स्थिरांक दशांश * src_state = s_in;

        स्थिरांक int32_t attn_stride_t = S_v * H;
        दशांश *       attn_ptr      = attn_out_base + (seq * n_tokens * H + head) * S_v;

        स्थिरांक दशांश zero = 0.0f;

        साठी (int32_t t = 0; t < n_tokens; t++) {
            स्थिरांक दशांश * q_t      = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) q + seq_q * q_nb3 + t * q_nb2 + h_q * q_nb1);
            स्थिरांक दशांश * k_t      = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) k + seq_k * k_nb3 + t * k_nb2 + h_k * k_nb1);
            स्थिरांक दशांश * v_t      = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) v + seq * v_nb3 + t * v_nb2 + head * v_nb1);
            स्थिरांक दशांश * g_t      = g + seq * g_stride_s + t * g_stride_t + head * g_stride_h;
            स्थिरांक दशांश   beta_val = beta[seq * b_stride_s + t * b_stride_t + head];

            // Precompute per-element gate for the kda path; scalar decay
            // otherwise. Decay is fused into per-j pass A/B below, not
            // applied to state in a separate pre-pass.
            दशांश decay = 0.0f;  // only used when !kda
            जर (kda) {
                स्थिरांक दशांश log2e = 1.4426950408889634f;
                __asm__ अस्थिर("mov.m.x m0, x0, 255\n" :::);
                __asm__ अस्थिर("fbc.ps f20, %[l2e]\n" : : [l2e] "m"(log2e) : "f20");
                साठी (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ अस्थिर(
                        "flw.ps f10, %[g_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fexp.ps f10, f10\n"
                        "fsw.ps f10, %[out]\n"
                        : [out] "=m"(*(दशांश (*)[8]) & exp_g_buf[i])
                        : [g_vec] "m"(*(स्थिरांक दशांश (*)[8]) & g_t[i])
                        : "f10");
                }
                __asm__ अस्थिर("mova.m.x %[ms]\n" : : [ms] "r"(default_mask));
            } अन्यथा {
                decay = et_expf(g_t[0]);
            }

            साठी (int32_t j = j_start; j < j_end; j++) {
                स्थिरांक दशांश * src_row = src_state + j * S_v;
                दशांश *       dst_row = s_out + j * S_v;

                __asm__ अस्थिर("mov.m.x m0, x0, 255\n" :::);
                जर (kda) {
                    __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
                    साठी (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(स्थिरांक दशांश (*)[8]) & src_row[i]),
                              [g_vec] "m"(*(स्थिरांक दशांश (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(स्थिरांक दशांश (*)[8]) & k_t[i])
                            : "f10", "f11", "f12", "f13");
                    }
                } अन्यथा {
                    __asm__ अस्थिर(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [d] "m"(decay)
                        : "f10", "f22");
                    साठी (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(स्थिरांक दशांश (*)[8]) & src_row[i]), [k_vec] "m"(*(स्थिरांक दशांश (*)[8]) &
                                                                                           k_t[i])
                            : "f10", "f11", "f13");
                    }
                }

                दशांश dot_sk = hsum_f10();
                __asm__ अस्थिर("mova.m.x %[ms]\n" : : [ms] "r"(default_mask));

                दशांश delta_j = (v_t[j] - dot_sk) * beta_val;

                // -------- Pass B: decay + outer product + attn --------
                __asm__ अस्थिर("mov.m.x m0, x0, 255\n" :::);
                जर (kda) {
                    __asm__ अस्थिर(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j)
                        : "f10", "f21");
                    साठी (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_out] "=m"(*(दशांश (*)[8]) & dst_row[i])
                            : [s_vec] "m"(*(स्थिरांक दशांश (*)[8]) & src_row[i]),
                              [g_vec] "m"(*(स्थिरांक दशांश (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(स्थिरांक दशांश (*)[8]) & k_t[i]), [q_vec] "m"(*(स्थिरांक दशांश (*)[8]) & q_t[i])
                            : "f10", "f11", "f12", "f13", "f14");
                    }
                } अन्यथा {
                    __asm__ अस्थिर(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j), [d] "m"(decay)
                        : "f10", "f21", "f22");
                    साठी (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_out] "=m"(*(दशांश (*)[8]) & dst_row[i])
                            : [s_vec] "m"(*(स्थिरांक दशांश (*)[8]) & src_row[i]),
                              [k_vec] "m"(*(स्थिरांक दशांश (*)[8]) & k_t[i]), [q_vec] "m"(*(स्थिरांक दशांश (*)[8]) & q_t[i])
                            : "f10", "f11", "f13", "f14");
                    }
                }

                दशांश attn_val = hsum_f10();
                __asm__ अस्थिर("mova.m.x %[ms]\n" : : [ms] "r"(default_mask));

                attn_ptr[j] = attn_val * scale;
            }

            // n-way merge snapshot: live state lives in slot 0 (== s_out).
            // Copies state to target snapshot slots [1, K-1] in reverse chronological order.
            // target_slot == 0 is the live buffer itself => no copy.
            // target_slot >= K (when n_tokens > K) => older slots are discarded.
            जर (K > 1) {
                स्थिरांक int32_t target_slot = (n_tokens - 1) - t;
                जर (target_slot > 0 && target_slot < K) {
                    दशांश * snap = state_out_base + target_slot * state_plane_floats + head_state_off;
                    साठी (int32_t j = j_start; j < j_end; j++) {
                        स्थिरांक दशांश * src = s_out + j * S_v;
                        दशांश *       dst = snap + j * S_v;
                        साठी (int32_t i = 0; i < S_v; i++) {
                            dst[i] = src[i];
                        }
                    }
                }
            }

            // After t=0, state lives in s_out; flip src_state so subsequent
            // timesteps read-modify-write in place.
            src_state = s_out;
            attn_ptr += attn_stride_t;
        }
    }

    परत 0;
}
