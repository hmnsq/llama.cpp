#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_ssm_अवलोकनम्_params {
    संरचना ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    संरचना ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    संरचना ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    संरचना ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    संरचना ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    संरचना ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    संरचना ggml_tensor src6;  // ids: [n_seqs] i32
    संरचना ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

स्थिर अन्तःस्थ भिन्न softplus_f32(भिन्न x) {
    प्रत्यागम x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_ssm_अवलोकनम्_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    ध्रुव पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    ध्रुव पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * src2 = &params->src2;
    संरचना ggml_tensor * src3 = &params->src3;
    संरचना ggml_tensor * src4 = &params->src4;
    संरचना ggml_tensor * src5 = &params->src5;
    संरचना ggml_tensor * src6 = &params->src6;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न *   s_दत्त   = (ध्रुव भिन्न *) src0->दत्त;
    ध्रुव भिन्न *   x_दत्त   = (ध्रुव भिन्न *) src1->दत्त;
    ध्रुव भिन्न *   dt_दत्त  = (ध्रुव भिन्न *) src2->दत्त;
    ध्रुव भिन्न *   A_दत्त   = (ध्रुव भिन्न *) src3->दत्त;
    ध्रुव भिन्न *   B_दत्त   = (ध्रुव भिन्न *) src4->दत्त;
    ध्रुव भिन्न *   C_दत्त   = (ध्रुव भिन्न *) src5->दत्त;
    ध्रुव int32_t * ids      = (ध्रुव int32_t *) src6->दत्त;
    भिन्न *         dst_दत्त = (भिन्न *) dst->दत्त;

    यदि (!s_दत्त || !x_दत्त || !dt_दत्त || !A_दत्त || !B_दत्त || !C_दत्त || !ids || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t d_state      = src0->ne[0];
    ध्रुव int64_t शीर्ष_dim     = src0->ne[1];
    ध्रुव int64_t n_शीर्ष       = src1->ne[1];
    ध्रुव int64_t n_गणः      = src4->ne[1];
    ध्रुव int64_t n_seq_tokens = src1->ne[2];
    ध्रुव int64_t n_seqs       = src1->ne[3];
    ध्रुव int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    ध्रुव int64_t K            = params->K;

    यदि (src0->nb[0] != परिमाण(भिन्न) || src1->nb[0] != परिमाण(भिन्न) || src2->nb[0] != परिमाण(भिन्न) ||
        src3->nb[0] != परिमाण(भिन्न) || src4->nb[0] != परिमाण(भिन्न) || src5->nb[0] != परिमाण(भिन्न) ||
        src6->nb[0] != परिमाण(int32_t) || dst->nb[0] != परिमाण(भिन्न)) {
        प्रत्यागम -1;
    }

    यदि (K < 1 || n_गणः <= 0 || n_शीर्ष % n_गणः != 0) {
        प्रत्यागम -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    ध्रुव int64_t dst_lanes_प्रत्येकम्_cl    = 16;
    ध्रुव int64_t heads_प्रत्येकम्_cacheline = शीर्ष_dim >= dst_lanes_प्रत्येकम्_cl ? 1 : (dst_lanes_प्रत्येकम्_cl / शीर्ष_dim);
    ध्रुव int64_t heads_प्रत्येकम्_खण्डम्     = heads_प्रत्येकम्_cacheline > 0 ? heads_प्रत्येकम्_cacheline : 1;
    ध्रुव int64_t खण्डानि_प्रत्येकम्_seq      = (n_शीर्ष + heads_प्रत्येकम्_खण्डम् - 1) / heads_प्रत्येकम्_खण्डम्;
    ध्रुव int64_t dim_chunk_lanes     = शीर्ष_dim >= dst_lanes_प्रत्येकम्_cl ? dst_lanes_प्रत्येकम्_cl : शीर्ष_dim;
    ध्रुव int64_t dim_chunks_प्रत्येकम्_शीर्ष = (शीर्ष_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    ध्रुव int64_t units_प्रत्येकम्_seq    = खण्डानि_प्रत्येकम्_seq * dim_chunks_प्रत्येकम्_शीर्ष;
    ध्रुव int64_t योग_units      = n_seqs * units_प्रत्येकम्_seq;
    ध्रुव int64_t units_प्रत्येकम्_thread = (योग_units + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
    ध्रुव int64_t unit_begin       = (int64_t) thread_id * units_प्रत्येकम्_thread;
    int64_t       unit_अन्तः         = unit_begin + units_प्रत्येकम्_thread;

    यदि (unit_begin >= योग_units) {
        प्रत्यागम 0;
    }

    यदि (unit_अन्तः > योग_units) {
        unit_अन्तः = योग_units;
    }

    ध्रुव पूर्णाङ्क     A_प्रसारणम् = (src3->ne[0] == 1);
    ध्रुव int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    ध्रुव भिन्न   log2e_const = 1.4426950408889634f;

    प्रति (int64_t unit = unit_begin; unit < unit_अन्तः; ++unit) {
        ध्रुव int64_t seq_idx       = unit / units_प्रत्येकम्_seq;
        ध्रुव int64_t unit_अन्तर्_seq   = unit % units_प्रत्येकम्_seq;
        ध्रुव int64_t खण्डम्_अन्तर्_seq  = unit_अन्तर्_seq / dim_chunks_प्रत्येकम्_शीर्ष;
        ध्रुव int64_t dim_chunk_idx = unit_अन्तर्_seq % dim_chunks_प्रत्येकम्_शीर्ष;
        ध्रुव int64_t शीर्ष_begin    = खण्डम्_अन्तर्_seq * heads_प्रत्येकम्_खण्डम्;
        int64_t       शीर्ष_अन्तः      = शीर्ष_begin + heads_प्रत्येकम्_खण्डम्;

        यदि (शीर्ष_अन्तः > n_शीर्ष) {
            शीर्ष_अन्तः = n_शीर्ष;
        }

        ध्रुव int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_अन्तः   = dim_begin + dim_chunk_lanes;
        यदि (dim_अन्तः > शीर्ष_dim) {
            dim_अन्तः = शीर्ष_dim;
        }

        ध्रुव int32_t state_seq = ids[seq_idx];

        प्रति (int64_t शीर्ष_idx = शीर्ष_begin; शीर्ष_idx < शीर्ष_अन्तः; ++शीर्ष_idx) {
            ध्रुव int64_t गणः_idx = शीर्ष_idx / (n_शीर्ष / n_गणः);

            // A pointer for this head: contiguous over state_idx when not broadcast
            ध्रुव भिन्न * A_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) A_दत्त + (size_t) शीर्ष_idx * src3->nb[1]);

            प्रति (int64_t dim_idx = dim_begin; dim_idx < dim_अन्तः; ++dim_idx) {
                ध्रुव भिन्न * state_src =
                    (ध्रुव भिन्न *) ((ध्रुव वर्ण *) s_दत्त + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) शीर्ष_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                भिन्न * state_dst =
                    (भिन्न *) ((वर्ण *) dst_दत्त + (size_t) y_elems * परिमाण(भिन्न) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) शीर्ष_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                प्रति (int64_t token_idx = 0; token_idx < n_seq_tokens; ++token_idx) {
                    ध्रुव भिन्न * x_ptr =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) x_दत्त + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) शीर्ष_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    ध्रुव भिन्न * dt_ptr =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) dt_दत्त + (size_t) शीर्ष_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    ध्रुव भिन्न * B_row =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) B_दत्त + (size_t) गणः_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    ध्रुव भिन्न * C_row =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) C_दत्त + (size_t) गणः_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    ध्रुव भिन्न dt_softplus = softplus_f32(*dt_ptr);
                    ध्रुव भिन्न x_dt        = (*x_ptr) * dt_softplus;
                    ध्रुव भिन्न dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    ध्रुव भिन्न * पूर्व_row = (token_idx == 0) ? state_src : state_dst;

                    भिन्न   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    यदि (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        अचिह्नित दीर्घ saved_आवरणम्;
                        __asm__ चञ्चल("mova.x.m %0" : "=r"(saved_आवरणम्));
                        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ चञ्चल(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        यदि (A_प्रसारणम्) {
                            // dA is a per-head scalar — compute once and splat.
                            ध्रुव भिन्न dA_scalar = et_expf(dt_softplus * (*A_row));
                            __asm__ चञ्चल("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        प्रति (; state_idx < d_state_vec; state_idx += 8) {
                            यदि (!A_प्रसारणम्) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ चञ्चल(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(ध्रुव भिन्न (*)[8]) & A_row[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ चञ्चल(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(भिन्न (*)[8]) & state_dst[state_idx])
                                : [पूर्व] "m"(*(ध्रुव भिन्न (*)[8]) & पूर्व_row[state_idx]),
                                  [bv] "m"(*(ध्रुव भिन्न (*)[8]) & B_row[state_idx]),
                                  [cv] "m"(*(ध्रुव भिन्न (*)[8]) & C_row[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ चञ्चल(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_आवरणम्));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    प्रति (; state_idx < d_state; ++state_idx) {
                        ध्रुव भिन्न पूर्व_state = पूर्व_row[state_idx];
                        ध्रुव भिन्न A_val      = A_प्रसारणम् ? *A_row : A_row[state_idx];
                        ध्रुव भिन्न dA         = et_expf(dt_softplus * A_val);
                        ध्रुव भिन्न st         = पूर्व_state * dA + B_row[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_row[state_idx];
                    }

                    ध्रुव int64_t slot = n_seq_tokens - 1 - token_idx;
                    यदि (slot > 0 && slot < K) {
                        भिन्न * state_snapshot =
                            (भिन्न *) ((वर्ण *) state_dst + (size_t) slot * n_seqs * src0->nb[3]);
                        प्रति (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_दत्त[seq_idx * (n_seq_tokens * n_शीर्ष * शीर्ष_dim) + token_idx * (n_शीर्ष * शीर्ष_dim) +
                             शीर्ष_idx * शीर्ष_dim + dim_idx] = sumf;
                }
            }
        }
    }

    प्रत्यागम 0;
}
