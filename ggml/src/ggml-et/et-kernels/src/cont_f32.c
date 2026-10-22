//******************************************************************************
// Bare Metal CONT F32 Kernel
// Converts non-contiguous tensors to contiguous memory layout
//
// Fast path:    src contiguous: flat vectorized copy by cache lines
// Aligned path: nb00==4 and ne00 % 16 == 0: distribute rows, no coherency issue
// Unaligned:    nb00==4 and ne00 not aligned: distribute by cache lines,
//               reverse-compute src coords, handle partial rows at boundaries
// Fallback:     nb00 != 4: scalar per-element
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_cont_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input tensor (non-contiguous)
    ਬਣਤਰ ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_copy_f32(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int32_t n) {
    int32_t       i       = 0;
    ਅਚਲ int32_t vec_ਸਿਰਾ = (n / 8) * 8;
    ਲਈ (; i < vec_ਸਿਰਾ; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])
            : [s] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src[i])
            : "f10");
    }
    ਲਈ (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ scalar_copy_f32(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int32_t n) {
    ਲਈ (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_cont_params * params, ਖਾਲੀ * env) {
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
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    ਅਚਲ int64_t nb00 = src0->nb[0];
    ਅਚਲ int64_t nb01 = src0->nb[1];
    ਅਚਲ int64_t nb02 = src0->nb[2];
    ਅਚਲ int64_t nb03 = src0->nb[3];

    ਅਚਲ int64_t ਕੁੱਲ_elements = ne00 * ne01 * ne02 * ne03;

    ਜੇ (ਕੁੱਲ_elements == 0) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    ਜੇ (src_contiguous) {
        ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl = 16;
        ਅਚਲ int64_t ਕੁੱਲ_cl     = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

        ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cl + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = thread_id * cl_ਪ੍ਰਤੀ_thread;
        int64_t       cl_ਸਿਰਾ        = cl_ਸ਼ੁਰੂ + cl_ਪ੍ਰਤੀ_thread;
        ਜੇ (cl_ਸਿਰਾ > ਕੁੱਲ_cl) {
            cl_ਸਿਰਾ = ਕੁੱਲ_cl;
        }
        ਜੇ (cl_ਸ਼ੁਰੂ >= ਕੁੱਲ_cl) {
            ਵਾਪਸ 0;
        }

        ਅਚਲ int64_t es = cl_ਸ਼ੁਰੂ * elems_ਪ੍ਰਤੀ_cl;
        int64_t       ee = cl_ਸਿਰਾ * elems_ਪ੍ਰਤੀ_cl;
        ਜੇ (ee > ਕੁੱਲ_elements) {
            ee = ਕੁੱਲ_elements;
        }

        vec_copy_f32(dst_ਅੰਕੜੇ + es, src0_ਅੰਕੜੇ + es, (int32_t) (ee - es));
        ਵਾਪਸ 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    ਜੇ (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl = 16;
        ਅਚਲ int64_t ਕੁੱਲ_cl     = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

        ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cl + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = thread_id * cl_ਪ੍ਰਤੀ_thread;
        int64_t       cl_ਸਿਰਾ        = cl_ਸ਼ੁਰੂ + cl_ਪ੍ਰਤੀ_thread;
        ਜੇ (cl_ਸਿਰਾ > ਕੁੱਲ_cl) {
            cl_ਸਿਰਾ = ਕੁੱਲ_cl;
        }
        ਜੇ (cl_ਸ਼ੁਰੂ >= ਕੁੱਲ_cl) {
            ਵਾਪਸ 0;
        }

        ਅਚਲ int64_t es = cl_ਸ਼ੁਰੂ * elems_ਪ੍ਰਤੀ_cl;
        int64_t       ee = cl_ਸਿਰਾ * elems_ਪ੍ਰਤੀ_cl;
        ਜੇ (ee > ਕੁੱਲ_elements) {
            ee = ਕੁੱਲ_elements;
        }

        ਲਈ (int64_t idx = es; idx < ee; idx++) {
            ਅਚਲ int64_t i00  = idx % ne00;
            ਅਚਲ int64_t rem1 = idx / ne00;
            ਅਚਲ int64_t i01  = rem1 % ne01;
            ਅਚਲ int64_t rem2 = rem1 / ne01;
            ਅਚਲ int64_t i02  = rem2 % ne02;
            ਅਚਲ int64_t i03  = rem2 / ne02;

            ਅਚਲ ਦਸ਼ਮਲਵ * sp =
                (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_ਅੰਕੜੇ[idx] = *sp;
        }
        ਵਾਪਸ 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    ਜੇ (ne00 % 16 == 0) {
        ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ      = ne01 * ne02 * ne03;
        ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ       = thread_id * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
        ਅਚਲ int64_t ਸਿਰਾ_ਪੰਕਤੀ = (ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread < ਕੁੱਲ_ਪੰਕਤੀਆਂ) ? (ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread) : ਕੁੱਲ_ਪੰਕਤੀਆਂ;

        ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            ਵਾਪਸ 0;
        }

        ਲਈ (int64_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ir++) {
            ਅਚਲ int64_t i03 = ir / (ne02 * ne01);
            ਅਚਲ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ਅਚਲ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + ir * ne00;

            vec_copy_f32(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, (int32_t) ne00);
        }
        ਵਾਪਸ 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl = 16;
        ਅਚਲ int64_t ਕੁੱਲ_cl     = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

        ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cl + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = thread_id * cl_ਪ੍ਰਤੀ_thread;
        int64_t       cl_ਸਿਰਾ        = cl_ਸ਼ੁਰੂ + cl_ਪ੍ਰਤੀ_thread;
        ਜੇ (cl_ਸਿਰਾ > ਕੁੱਲ_cl) {
            cl_ਸਿਰਾ = ਕੁੱਲ_cl;
        }
        ਜੇ (cl_ਸ਼ੁਰੂ >= ਕੁੱਲ_cl) {
            ਵਾਪਸ 0;
        }

        ਅਚਲ int64_t es = cl_ਸ਼ੁਰੂ * elems_ਪ੍ਰਤੀ_cl;
        int64_t       ee = cl_ਸਿਰਾ * elems_ਪ੍ਰਤੀ_cl;
        ਜੇ (ee > ਕੁੱਲ_elements) {
            ee = ਕੁੱਲ_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t ਪੰਕਤੀ_idx = pos / ne00;
        int64_t ਖਾਨਾ     = pos % ne00;

        ਜਦੋਂ (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            ਅਚਲ int64_t i03 = ਪੰਕਤੀ_idx / (ne02 * ne01);
            ਅਚਲ int64_t i02 = (ਪੰਕਤੀ_idx - i03 * ne02 * ne01) / ne01;
            ਅਚਲ int64_t i01 = ਪੰਕਤੀ_idx - i03 * ne02 * ne01 - i02 * ne01;

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t ਪੰਕਤੀ_ਬਚਿਆ   = ne00 - ਖਾਨਾ;
            int64_t chunk_ਬਚਿਆ = ee - pos;
            int32_t n               = (int32_t) (ਪੰਕਤੀ_ਬਚਿਆ < chunk_ਬਚਿਆ ? ਪੰਕਤੀ_ਬਚਿਆ : chunk_ਬਚਿਆ);

            vec_copy_f32(dst_ਅੰਕੜੇ + pos, src_ਪੰਕਤੀ + ਖਾਨਾ, n);

            pos += n;
            ਖਾਨਾ = 0;  // subsequent rows start at column 0
            ਪੰਕਤੀ_idx++;
        }
    }

    ਵਾਪਸ 0;
}
