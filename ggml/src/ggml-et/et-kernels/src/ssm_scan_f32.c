#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_ssm_scan_params {
    काष्ठा ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    काष्ठा ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    काष्ठा ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    काष्ठा ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    काष्ठा ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    काष्ठा ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    काष्ठा ggml_tensor src6;  // ids: [n_seqs] i32
    काष्ठा ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

अटल अंतरभूत भग्न softplus_f32(भग्न x) {
    वापस x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

पूर्णांक entry_point(काष्ठा ggml_et_ssm_scan_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    स्थिर पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    स्थिर पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * src2 = &params->src2;
    काष्ठा ggml_tensor * src3 = &params->src3;
    काष्ठा ggml_tensor * src4 = &params->src4;
    काष्ठा ggml_tensor * src5 = &params->src5;
    काष्ठा ggml_tensor * src6 = &params->src6;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न *   s_तथ्य   = (स्थिर भग्न *) src0->तथ्य;
    स्थिर भग्न *   x_तथ्य   = (स्थिर भग्न *) src1->तथ्य;
    स्थिर भग्न *   dt_तथ्य  = (स्थिर भग्न *) src2->तथ्य;
    स्थिर भग्न *   A_तथ्य   = (स्थिर भग्न *) src3->तथ्य;
    स्थिर भग्न *   B_तथ्य   = (स्थिर भग्न *) src4->तथ्य;
    स्थिर भग्न *   C_तथ्य   = (स्थिर भग्न *) src5->तथ्य;
    स्थिर int32_t * ids      = (स्थिर int32_t *) src6->तथ्य;
    भग्न *         dst_तथ्य = (भग्न *) dst->तथ्य;

    अगर (!s_तथ्य || !x_तथ्य || !dt_तथ्य || !A_तथ्य || !B_तथ्य || !C_तथ्य || !ids || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t d_state      = src0->ne[0];
    स्थिर int64_t शीर्ष_dim     = src0->ne[1];
    स्थिर int64_t n_शीर्ष       = src1->ne[1];
    स्थिर int64_t n_group      = src4->ne[1];
    स्थिर int64_t n_seq_tokens = src1->ne[2];
    स्थिर int64_t n_seqs       = src1->ne[3];
    स्थिर int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    स्थिर int64_t K            = params->K;

    अगर (src0->nb[0] != माप(भग्न) || src1->nb[0] != माप(भग्न) || src2->nb[0] != माप(भग्न) ||
        src3->nb[0] != माप(भग्न) || src4->nb[0] != माप(भग्न) || src5->nb[0] != माप(भग्न) ||
        src6->nb[0] != माप(int32_t) || dst->nb[0] != माप(भग्न)) {
        वापस -1;
    }

    अगर (K < 1 || n_group <= 0 || n_शीर्ष % n_group != 0) {
        वापस -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    स्थिर int64_t dst_lanes_per_cl    = 16;
    स्थिर int64_t heads_per_cacheline = शीर्ष_dim >= dst_lanes_per_cl ? 1 : (dst_lanes_per_cl / शीर्ष_dim);
    स्थिर int64_t heads_per_block     = heads_per_cacheline > 0 ? heads_per_cacheline : 1;
    स्थिर int64_t blocks_per_seq      = (n_शीर्ष + heads_per_block - 1) / heads_per_block;
    स्थिर int64_t dim_chunk_lanes     = शीर्ष_dim >= dst_lanes_per_cl ? dst_lanes_per_cl : शीर्ष_dim;
    स्थिर int64_t dim_chunks_per_शीर्ष = (शीर्ष_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    स्थिर int64_t units_per_seq    = blocks_per_seq * dim_chunks_per_शीर्ष;
    स्थिर int64_t योग_units      = n_seqs * units_per_seq;
    स्थिर int64_t units_per_thread = (योग_units + अंक_threads - 1) / अंक_threads;
    स्थिर int64_t unit_begin       = (int64_t) thread_id * units_per_thread;
    int64_t       unit_end         = unit_begin + units_per_thread;

    अगर (unit_begin >= योग_units) {
        वापस 0;
    }

    अगर (unit_end > योग_units) {
        unit_end = योग_units;
    }

    स्थिर पूर्णांक     A_broadcast = (src3->ne[0] == 1);
    स्थिर int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    स्थिर भग्न   log2e_const = 1.4426950408889634f;

    क्रम (int64_t unit = unit_begin; unit < unit_end; ++unit) {
        स्थिर int64_t seq_idx       = unit / units_per_seq;
        स्थिर int64_t unit_in_seq   = unit % units_per_seq;
        स्थिर int64_t block_in_seq  = unit_in_seq / dim_chunks_per_शीर्ष;
        स्थिर int64_t dim_chunk_idx = unit_in_seq % dim_chunks_per_शीर्ष;
        स्थिर int64_t शीर्ष_begin    = block_in_seq * heads_per_block;
        int64_t       शीर्ष_end      = शीर्ष_begin + heads_per_block;

        अगर (शीर्ष_end > n_शीर्ष) {
            शीर्ष_end = n_शीर्ष;
        }

        स्थिर int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_end   = dim_begin + dim_chunk_lanes;
        अगर (dim_end > शीर्ष_dim) {
            dim_end = शीर्ष_dim;
        }

        स्थिर int32_t state_seq = ids[seq_idx];

        क्रम (int64_t शीर्ष_idx = शीर्ष_begin; शीर्ष_idx < शीर्ष_end; ++शीर्ष_idx) {
            स्थिर int64_t group_idx = शीर्ष_idx / (n_शीर्ष / n_group);

            // A pointer for this head: contiguous over state_idx when not broadcast
            स्थिर भग्न * A_row = (स्थिर भग्न *) ((स्थिर अक्षर *) A_तथ्य + (माप_प्रकार) शीर्ष_idx * src3->nb[1]);

            क्रम (int64_t dim_idx = dim_begin; dim_idx < dim_end; ++dim_idx) {
                स्थिर भग्न * state_src =
                    (स्थिर भग्न *) ((स्थिर अक्षर *) s_तथ्य + (माप_प्रकार) dim_idx * src0->nb[1] +
                                     (माप_प्रकार) शीर्ष_idx * src0->nb[2] + (माप_प्रकार) state_seq * src0->nb[3]);

                भग्न * state_dst =
                    (भग्न *) ((अक्षर *) dst_तथ्य + (माप_प्रकार) y_elems * माप(भग्न) + (माप_प्रकार) dim_idx * src0->nb[1] +
                               (माप_प्रकार) शीर्ष_idx * src0->nb[2] + (माप_प्रकार) seq_idx * src0->nb[3]);

                क्रम (int64_t token_idx = 0; token_idx < n_seq_tokens; ++token_idx) {
                    स्थिर भग्न * x_ptr =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) x_तथ्य + (माप_प्रकार) dim_idx * src1->nb[0] +
                                         (माप_प्रकार) शीर्ष_idx * src1->nb[1] + (माप_प्रकार) token_idx * src1->nb[2] +
                                         (माप_प्रकार) seq_idx * src1->nb[3]);

                    स्थिर भग्न * dt_ptr =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) dt_तथ्य + (माप_प्रकार) शीर्ष_idx * src2->nb[0] +
                                         (माप_प्रकार) token_idx * src2->nb[1] + (माप_प्रकार) seq_idx * src2->nb[2]);

                    स्थिर भग्न * B_row =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) B_तथ्य + (माप_प्रकार) group_idx * src4->nb[1] +
                                         (माप_प्रकार) token_idx * src4->nb[2] + (माप_प्रकार) seq_idx * src4->nb[3]);

                    स्थिर भग्न * C_row =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) C_तथ्य + (माप_प्रकार) group_idx * src5->nb[1] +
                                         (माप_प्रकार) token_idx * src5->nb[2] + (माप_प्रकार) seq_idx * src5->nb[3]);

                    स्थिर भग्न dt_softplus = softplus_f32(*dt_ptr);
                    स्थिर भग्न x_dt        = (*x_ptr) * dt_softplus;
                    स्थिर भग्न dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    स्थिर भग्न * पूर्व_row = (token_idx == 0) ? state_src : state_dst;

                    भग्न   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    अगर (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        अचिन्हित दीर्घ saved_mask;
                        __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
                        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ अस्थिर(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        अगर (A_broadcast) {
                            // dA is a per-head scalar — compute once and splat.
                            स्थिर भग्न dA_scalar = et_expf(dt_softplus * (*A_row));
                            __asm__ अस्थिर("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        क्रम (; state_idx < d_state_vec; state_idx += 8) {
                            अगर (!A_broadcast) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ अस्थिर(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(स्थिर भग्न (*)[8]) & A_row[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ अस्थिर(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(भग्न (*)[8]) & state_dst[state_idx])
                                : [पूर्व] "m"(*(स्थिर भग्न (*)[8]) & पूर्व_row[state_idx]),
                                  [bv] "m"(*(स्थिर भग्न (*)[8]) & B_row[state_idx]),
                                  [cv] "m"(*(स्थिर भग्न (*)[8]) & C_row[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ अस्थिर(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    क्रम (; state_idx < d_state; ++state_idx) {
                        स्थिर भग्न पूर्व_state = पूर्व_row[state_idx];
                        स्थिर भग्न A_val      = A_broadcast ? *A_row : A_row[state_idx];
                        स्थिर भग्न dA         = et_expf(dt_softplus * A_val);
                        स्थिर भग्न st         = पूर्व_state * dA + B_row[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_row[state_idx];
                    }

                    स्थिर int64_t slot = n_seq_tokens - 1 - token_idx;
                    अगर (slot > 0 && slot < K) {
                        भग्न * state_snapshot =
                            (भग्न *) ((अक्षर *) state_dst + (माप_प्रकार) slot * n_seqs * src0->nb[3]);
                        क्रम (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_तथ्य[seq_idx * (n_seq_tokens * n_शीर्ष * शीर्ष_dim) + token_idx * (n_शीर्ष * शीर्ष_dim) +
                             शीर्ष_idx * शीर्ष_dim + dim_idx] = sumf;
                }
            }
        }
    }

    वापस 0;
}
