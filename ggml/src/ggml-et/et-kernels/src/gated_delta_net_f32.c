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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_gated_delta_net_params {
    கட்டமைப்பு ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    கட்டமைப்பு ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    கட்டமைப்பு ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    கட்டமைப்பு ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    கட்டமைப்பு ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    கட்டமைப்பு ggml_tensor state_in;  // [S_v*S_v*H, K, n_seqs]
    கட்டமைப்பு ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
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
    தசம              scale;     // 1/sqrt(S_v)
};

நிலையான உள்வரி தசம hsum_f10(வெறுமை) {
    தசம result;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    திரும்பு result;
}

முழு entry_point(கட்டமைப்பு ggml_et_gated_delta_net_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    மாறிலி கட்டமைப்பு ggml_tensor * q_tsr     = &params->q;
    மாறிலி கட்டமைப்பு ggml_tensor * k_tsr     = &params->k;
    மாறிலி கட்டமைப்பு ggml_tensor * v_tsr     = &params->v;
    மாறிலி கட்டமைப்பு ggml_tensor * g_tsr     = &params->g;
    மாறிலி கட்டமைப்பு ggml_tensor * beta_tsr  = &params->beta;
    மாறிலி கட்டமைப்பு ggml_tensor * state_tsr = &params->state_in;
    மாறிலி கட்டமைப்பு ggml_tensor * dst_tsr   = &params->dst;

    மாறிலி தசம * q        = (மாறிலி தசம *) q_tsr->data;
    மாறிலி தசம * k        = (மாறிலி தசம *) k_tsr->data;
    மாறிலி தசம * v        = (மாறிலி தசம *) v_tsr->data;
    மாறிலி தசம * g        = (மாறிலி தசம *) g_tsr->data;
    மாறிலி தசம * beta     = (மாறிலி தசம *) beta_tsr->data;
    மாறிலி தசம * state_in = (மாறிலி தசம *) state_tsr->data;
    தசம *       dst_data = (தசம *) dst_tsr->data;

    மாறிலி int32_t S_v      = params->S_v;
    மாறிலி int32_t H        = params->H;
    மாறிலி int32_t H_q      = params->H_q;
    மாறிலி int32_t H_k      = params->H_k;
    மாறிலி int32_t n_tokens = params->n_tokens;
    மாறிலி int32_t n_seqs   = params->n_seqs;
    மாறிலி int32_t n_seqs_q = params->n_seqs_q;
    மாறிலி int32_t n_seqs_k = params->n_seqs_k;
    மாறிலி int32_t kda      = params->kda;
    மாறிலி int32_t K        = params->K;
    மாறிலி தசம   scale    = params->scale;

    எனில் (!q || !k || !v || !g || !beta || !state_in || !dst_data) {
        திரும்பு -1;
    }

    // Preserve the original contract for every tensor except q, k, and v, which may be
    // row-contiguous with strided higher dimensions.
    எனில் (q_tsr->nb[0] != அளவு(தசம) || k_tsr->nb[0] != அளவு(தசம) || v_tsr->nb[0] != அளவு(தசம) ||
        g_tsr->nb[0] != அளவு(தசம) || beta_tsr->nb[0] != அளவு(தசம) || state_tsr->nb[0] != அளவு(தசம) ||
        dst_tsr->nb[0] != அளவு(தசம)) {
        திரும்பு -1;
    }

    மாறிலி int32_t attn_elems     = S_v * H * n_tokens * n_seqs;
    தசம *       attn_out_base  = dst_data;
    தசம *       state_out_base = dst_data + attn_elems;

    மாறிலி int32_t state_plane_floats  = S_v * S_v * H * n_seqs;

    மாறிலி int32_t G0 = kda ? S_v : 1;

    மாறிலி size_t  q_nb1      = q_tsr->nb[1];
    மாறிலி size_t  q_nb2      = q_tsr->nb[2];
    மாறிலி size_t  q_nb3      = q_tsr->nb[3];
    மாறிலி size_t  k_nb1      = k_tsr->nb[1];
    மாறிலி size_t  k_nb2      = k_tsr->nb[2];
    மாறிலி size_t  k_nb3      = k_tsr->nb[3];
    மாறிலி size_t  v_nb1      = v_tsr->nb[1];
    மாறிலி size_t  v_nb2      = v_tsr->nb[2];
    மாறிலி size_t  v_nb3      = v_tsr->nb[3];
    மாறிலி int32_t g_stride_h = G0;
    மாறிலி int32_t g_stride_t = G0 * H;
    மாறிலி int32_t g_stride_s = G0 * H * n_tokens;
    மாறிலி int32_t b_stride_t = H;
    மாறிலி int32_t b_stride_s = H * n_tokens;

    தசம exp_g_buf[128];

    // FP and SIMD share the same register file. Scalar FP needs the default
    // mask; 8-wide .ps blocks need m0=255. Save once, toggle at boundaries.
    unsigned long default_mask;
    __asm__ நிலையற்ற("mova.x.m %[ms]\n" : [ms] "=r"(default_mask));

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
    மாறிலி int32_t J_BLK      = ET_CACHE_வரி_SIZE_BYTES / (int32_t) அளவு(தசம);
    மாறிலி int32_t n_j_blocks = (S_v + J_BLK - 1) / J_BLK;
    மாறிலி int32_t total_work = n_j_blocks * H * n_seqs;

    க்கு (int32_t ir = thread_id; ir < total_work; ir += இலக்கம்_threads) {
        மாறிலி int32_t jb   = ir % n_j_blocks;
        மாறிலி int32_t head = (ir / n_j_blocks) % H;
        மாறிலி int32_t seq  = ir / (n_j_blocks * H);

        மாறிலி int32_t j_start = jb * J_BLK;
        மாறிலி int32_t j_end   = (j_start + J_BLK < S_v) ? j_start + J_BLK : S_v;

        மாறிலி int32_t h_q   = head % H_q;
        மாறிலி int32_t h_k   = head % H_k;
        மாறிலி int32_t seq_q = (n_seqs_q == n_seqs) ? seq : (seq * n_seqs_q / n_seqs);
        மாறிலி int32_t seq_k = (n_seqs_k == n_seqs) ? seq : (seq * n_seqs_k / n_seqs);

        மாறிலி int32_t head_state_off = (seq * H + head) * S_v * S_v;
        // Live RMW buffer = first snapshot plane (slot 0).
        தசம *       s_out          = state_out_base + head_state_off;
        // Input state: seq `seq`, head `head`.
        மாறிலி தசம * s_in           = state_in + head_state_off;

        // Skip the explicit s_in -> s_out copy. At t=0 pass A/B read through
        // src_state = s_in; pass B writes the first new row to s_out. From
        // t=1 onward src_state flips to s_out (read-modify-write in place).
        மாறிலி தசம * src_state = s_in;

        மாறிலி int32_t attn_stride_t = S_v * H;
        தசம *       attn_ptr      = attn_out_base + (seq * n_tokens * H + head) * S_v;

        மாறிலி தசம zero = 0.0f;

        க்கு (int32_t t = 0; t < n_tokens; t++) {
            மாறிலி தசம * q_t      = (மாறிலி தசம *) ((மாறிலி எழுத்து *) q + seq_q * q_nb3 + t * q_nb2 + h_q * q_nb1);
            மாறிலி தசம * k_t      = (மாறிலி தசம *) ((மாறிலி எழுத்து *) k + seq_k * k_nb3 + t * k_nb2 + h_k * k_nb1);
            மாறிலி தசம * v_t      = (மாறிலி தசம *) ((மாறிலி எழுத்து *) v + seq * v_nb3 + t * v_nb2 + head * v_nb1);
            மாறிலி தசம * g_t      = g + seq * g_stride_s + t * g_stride_t + head * g_stride_h;
            மாறிலி தசம   beta_val = beta[seq * b_stride_s + t * b_stride_t + head];

            // Precompute per-element gate for the kda path; scalar decay
            // otherwise. Decay is fused into per-j pass A/B below, not
            // applied to state in a separate pre-pass.
            தசம decay = 0.0f;  // only used when !kda
            எனில் (kda) {
                மாறிலி தசம log2e = 1.4426950408889634f;
                __asm__ நிலையற்ற("mov.m.x m0, x0, 255\n" :::);
                __asm__ நிலையற்ற("fbc.ps f20, %[l2e]\n" : : [l2e] "m"(log2e) : "f20");
                க்கு (int32_t i = 0; i < S_v; i += 8) {
                    __asm__ நிலையற்ற(
                        "flw.ps f10, %[g_vec]\n"
                        "fmul.ps f10, f10, f20, rne\n"
                        "fexp.ps f10, f10\n"
                        "fsw.ps f10, %[out]\n"
                        : [out] "=m"(*(தசம (*)[8]) & exp_g_buf[i])
                        : [g_vec] "m"(*(மாறிலி தசம (*)[8]) & g_t[i])
                        : "f10");
                }
                __asm__ நிலையற்ற("mova.m.x %[ms]\n" : : [ms] "r"(default_mask));
            } இல்லையெனில் {
                decay = et_expf(g_t[0]);
            }

            க்கு (int32_t j = j_start; j < j_end; j++) {
                மாறிலி தசம * src_row = src_state + j * S_v;
                தசம *       dst_row = s_out + j * S_v;

                __asm__ நிலையற்ற("mov.m.x m0, x0, 255\n" :::);
                எனில் (kda) {
                    __asm__ நிலையற்ற("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
                    க்கு (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ நிலையற்ற(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(மாறிலி தசம (*)[8]) & src_row[i]),
                              [g_vec] "m"(*(மாறிலி தசம (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(மாறிலி தசம (*)[8]) & k_t[i])
                            : "f10", "f11", "f12", "f13");
                    }
                } இல்லையெனில் {
                    __asm__ நிலையற்ற(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [d] "m"(decay)
                        : "f10", "f22");
                    க்கு (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ நிலையற்ற(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f10, f11, f13, f10\n"  // acc += row_dec * k
                            :
                            : [s_vec] "m"(*(மாறிலி தசம (*)[8]) & src_row[i]), [k_vec] "m"(*(மாறிலி தசம (*)[8]) &
                                                                                           k_t[i])
                            : "f10", "f11", "f13");
                    }
                }

                தசம dot_sk = hsum_f10();
                __asm__ நிலையற்ற("mova.m.x %[ms]\n" : : [ms] "r"(default_mask));

                தசம delta_j = (v_t[j] - dot_sk) * beta_val;

                // -------- Pass B: decay + outer product + attn --------
                __asm__ நிலையற்ற("mov.m.x m0, x0, 255\n" :::);
                எனில் (kda) {
                    __asm__ நிலையற்ற(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j)
                        : "f10", "f21");
                    க்கு (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ நிலையற்ற(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f12, %[g_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f12\n"        // row_dec = row * g
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_out] "=m"(*(தசம (*)[8]) & dst_row[i])
                            : [s_vec] "m"(*(மாறிலி தசம (*)[8]) & src_row[i]),
                              [g_vec] "m"(*(மாறிலி தசம (*)[8]) & exp_g_buf[i]),
                              [k_vec] "m"(*(மாறிலி தசம (*)[8]) & k_t[i]), [q_vec] "m"(*(மாறிலி தசம (*)[8]) & q_t[i])
                            : "f10", "f11", "f12", "f13", "f14");
                    }
                } இல்லையெனில் {
                    __asm__ நிலையற்ற(
                        "fbc.ps f10, %[z]\n"
                        "fbc.ps f21, %[dj]\n"
                        "fbc.ps f22, %[d]\n"
                        :
                        : [z] "m"(zero), [dj] "m"(delta_j), [d] "m"(decay)
                        : "f10", "f21", "f22");
                    க்கு (int32_t i = 0; i < S_v; i += 8) {
                        __asm__ நிலையற்ற(
                            "flw.ps f11, %[s_vec]\n"
                            "flw.ps f13, %[k_vec]\n"
                            "flw.ps f14, %[q_vec]\n"
                            "fmul.ps f11, f11, f22\n"        // row_dec = row * decay
                            "fmadd.ps f11, f13, f21, f11\n"  // row_new = row_dec + k*delta_j
                            "fsw.ps f11, %[s_out]\n"
                            "fmadd.ps f10, f11, f14, f10\n"  // attn_acc += row_new * q
                            : [s_out] "=m"(*(தசம (*)[8]) & dst_row[i])
                            : [s_vec] "m"(*(மாறிலி தசம (*)[8]) & src_row[i]),
                              [k_vec] "m"(*(மாறிலி தசம (*)[8]) & k_t[i]), [q_vec] "m"(*(மாறிலி தசம (*)[8]) & q_t[i])
                            : "f10", "f11", "f13", "f14");
                    }
                }

                தசம attn_val = hsum_f10();
                __asm__ நிலையற்ற("mova.m.x %[ms]\n" : : [ms] "r"(default_mask));

                attn_ptr[j] = attn_val * scale;
            }

            // n-way merge snapshot: live state lives in slot 0 (== s_out).
            // Copies state to target snapshot slots [1, K-1] in reverse chronological order.
            // target_slot == 0 is the live buffer itself => no copy.
            // target_slot >= K (when n_tokens > K) => older slots are discarded.
            எனில் (K > 1) {
                மாறிலி int32_t target_slot = (n_tokens - 1) - t;
                எனில் (target_slot > 0 && target_slot < K) {
                    தசம * snap = state_out_base + target_slot * state_plane_floats + head_state_off;
                    க்கு (int32_t j = j_start; j < j_end; j++) {
                        மாறிலி தசம * src = s_out + j * S_v;
                        தசம *       dst = snap + j * S_v;
                        க்கு (int32_t i = 0; i < S_v; i++) {
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

    திரும்பு 0;
}
