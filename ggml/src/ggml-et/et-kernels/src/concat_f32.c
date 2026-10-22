//******************************************************************************
// Concat F32 Kernel
// Concatenates two F32 tensors along a specified dimension.
// All copies are aligned to cacheline boundaries (64 bytes = 16 floats).
//
// For dim >= 1, entire rows are copied from src0 or src1 into dst.
// For dim == 0, use:
// - a fast vector path when both source row segments are cacheline-aligned
// - a scalar stride-aware path otherwise
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

ਬਣਤਰ ggml_et_concat_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input tensor 0
    ਬਣਤਰ ggml_tensor src1;  // F32 input tensor 1
    ਬਣਤਰ ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ copy_ਪੰਕਤੀ_aligned(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int32_t n) {
    ਲਈ (int32_t i = 0; i < n; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src[i])
            : "f11");
    }
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_concat_params * params, ਖਾਲੀ * env) {
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

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ਅਚਲ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    ਅਚਲ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    ਅਚਲ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ਅਚਲ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    ਅਚਲ size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    ਜੇ (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || nb10 != ਆਕਾਰ(ਦਸ਼ਮਲਵ))) {
        ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ = et_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_cacheline_ਸਮੂਹ(ne0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ int64_t ਕੁੱਲ_groups   = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;

        ਲਈ (int64_t grp = thread_id; grp < ਕੁੱਲ_groups; grp += ਅੰਕ_ਧਾਗੇ) {
            ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = grp * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
            int64_t       ਪੰਕਤੀ_ਸਿਰਾ   = ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
            ਜੇ (ਪੰਕਤੀ_ਸਿਰਾ > ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
                ਪੰਕਤੀ_ਸਿਰਾ = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
            }

            ਲਈ (int64_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; ਪੰਕਤੀ++) {
                int64_t i1 = ਪੰਕਤੀ % ne1;
                int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
                int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);

                ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                ਅਚਲ ਅੱਖਰ * s0_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03;
                ਲਈ (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_ਪੰਕਤੀ[i0] = *(ਅਚਲ ਦਸ਼ਮਲਵ *) (s0_ਅਧਾਰ + i0 * nb00);
                }

                ਅਚਲ ਅੱਖਰ * s1_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + i1 * nb11 + i2 * nb12 + i3 * nb13;
                ਲਈ (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_ਪੰਕਤੀ[ne00 + i0] = *(ਅਚਲ ਦਸ਼ਮਲਵ *) (s1_ਅਧਾਰ + i0 * nb10);
                }
            }
        }
        ਵਾਪਸ 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    ਲਈ (int64_t ਪੰਕਤੀ = thread_id; ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ += ਅੰਕ_ਧਾਗੇ) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = ਪੰਕਤੀ % ne1;
        int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
        int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);

        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ਜੇ (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            ਅਚਲ ਦਸ਼ਮਲਵ * s0_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ਅਚਲ ਦਸ਼ਮਲਵ * s1_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, s0_ਪੰਕਤੀ, (int32_t) ne00);
            copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ + ne00, s1_ਪੰਕਤੀ, (int32_t) ne10);

        } ਨਹੀਂਤਾਂ ਜੇ (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            ਜੇ (i1 < ne01) {
                ਅਚਲ ਦਸ਼ਮਲਵ * s0_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, s0_ਪੰਕਤੀ, (int32_t) ne0);
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਦਸ਼ਮਲਵ * s1_ਪੰਕਤੀ =
                    (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, s1_ਪੰਕਤੀ, (int32_t) ne0);
            }

        } ਨਹੀਂਤਾਂ ਜੇ (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            ਜੇ (i2 < ne02) {
                ਅਚਲ ਦਸ਼ਮਲਵ * s0_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, s0_ਪੰਕਤੀ, (int32_t) ne0);
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਦਸ਼ਮਲਵ * s1_ਪੰਕਤੀ =
                    (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, s1_ਪੰਕਤੀ, (int32_t) ne0);
            }

        } ਨਹੀਂਤਾਂ {
            // dim == 3: first ne03 batches from src0, rest from src1
            ਜੇ (i3 < ne03) {
                ਅਚਲ ਦਸ਼ਮਲਵ * s0_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, s0_ਪੰਕਤੀ, (int32_t) ne0);
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ ਦਸ਼ਮਲਵ * s1_ਪੰਕਤੀ =
                    (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_ਪੰਕਤੀ_aligned(dst_ਪੰਕਤੀ, s1_ਪੰਕਤੀ, (int32_t) ne0);
            }
        }
    }

    ਵਾਪਸ 0;
}
