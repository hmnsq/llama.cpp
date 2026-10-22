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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_cont_params {
    ರಚನೆ ggml_tensor src0;  // F32 input tensor (non-contiguous)
    ರಚನೆ ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_copy_f32(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int32_t n) {
    int32_t       i       = 0;
    ಸ್ಥಿರಾಂಕ int32_t vec_ಕೊನೆ = (n / 8) * 8;
    ಗಾಗಿ (; i < vec_ಕೊನೆ; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst[i])
            : [s] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src[i])
            : "f10");
    }
    ಗಾಗಿ (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ scalar_copy_f32(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int32_t n) {
    ಗಾಗಿ (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_cont_params * params, ಶೂನ್ಯ * env) {
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

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ int64_t nb00 = src0->nb[0];
    ಸ್ಥಿರಾಂಕ int64_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb03 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = ne00 * ne01 * ne02 * ne03;

    ಆದರೆ (ಒಟ್ಟು_elements == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    ಆದರೆ (src_contiguous) {
        ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl = 16;
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl     = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

        ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cl + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
        int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
        ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cl) {
            cl_ಕೊನೆ = ಒಟ್ಟು_cl;
        }
        ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cl) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಸ್ಥಿರಾಂಕ int64_t es = cl_ಆರಂಭ * elems_ಪ್ರತಿ_cl;
        int64_t       ee = cl_ಕೊನೆ * elems_ಪ್ರತಿ_cl;
        ಆದರೆ (ee > ಒಟ್ಟು_elements) {
            ee = ಒಟ್ಟು_elements;
        }

        vec_copy_f32(dst_ದತ್ತಾಂಶ + es, src0_ದತ್ತಾಂಶ + es, (int32_t) (ee - es));
        ಹಿಂತಿರುಗಿ 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    ಆದರೆ (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl = 16;
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl     = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

        ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cl + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
        int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
        ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cl) {
            cl_ಕೊನೆ = ಒಟ್ಟು_cl;
        }
        ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cl) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಸ್ಥಿರಾಂಕ int64_t es = cl_ಆರಂಭ * elems_ಪ್ರತಿ_cl;
        int64_t       ee = cl_ಕೊನೆ * elems_ಪ್ರತಿ_cl;
        ಆದರೆ (ee > ಒಟ್ಟು_elements) {
            ee = ಒಟ್ಟು_elements;
        }

        ಗಾಗಿ (int64_t idx = es; idx < ee; idx++) {
            ಸ್ಥಿರಾಂಕ int64_t i00  = idx % ne00;
            ಸ್ಥಿರಾಂಕ int64_t rem1 = idx / ne00;
            ಸ್ಥಿರಾಂಕ int64_t i01  = rem1 % ne01;
            ಸ್ಥಿರಾಂಕ int64_t rem2 = rem1 / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i02  = rem2 % ne02;
            ಸ್ಥಿರಾಂಕ int64_t i03  = rem2 / ne02;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sp =
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_ದತ್ತಾಂಶ[idx] = *sp;
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    ಆದರೆ (ne00 % 16 == 0) {
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು      = ne01 * ne02 * ne03;
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು       = thread_id * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
        ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು = (ಆರಂಭ_ಅಡ್ಡಸಾಲು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) ? (ಆರಂಭ_ಅಡ್ಡಸಾಲು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread) : ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

        ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಗಾಗಿ (int64_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ir++) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne02 * ne01);
            ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + ir * ne00;

            vec_copy_f32(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, (int32_t) ne00);
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl = 16;
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl     = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

        ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cl + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
        int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
        ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cl) {
            cl_ಕೊನೆ = ಒಟ್ಟು_cl;
        }
        ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cl) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಸ್ಥಿರಾಂಕ int64_t es = cl_ಆರಂಭ * elems_ಪ್ರತಿ_cl;
        int64_t       ee = cl_ಕೊನೆ * elems_ಪ್ರತಿ_cl;
        ಆದರೆ (ee > ಒಟ್ಟು_elements) {
            ee = ಒಟ್ಟು_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t ಅಡ್ಡಸಾಲು_idx = pos / ne00;
        int64_t ಲಂಬಸಾಲು     = pos % ne00;

        ಆಗ (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            ಸ್ಥಿರಾಂಕ int64_t i03 = ಅಡ್ಡಸಾಲು_idx / (ne02 * ne01);
            ಸ್ಥಿರಾಂಕ int64_t i02 = (ಅಡ್ಡಸಾಲು_idx - i03 * ne02 * ne01) / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i01 = ಅಡ್ಡಸಾಲು_idx - i03 * ne02 * ne01 - i02 * ne01;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t ಅಡ್ಡಸಾಲು_ಉಳಿದ   = ne00 - ಲಂಬಸಾಲು;
            int64_t chunk_ಉಳಿದ = ee - pos;
            int32_t n               = (int32_t) (ಅಡ್ಡಸಾಲು_ಉಳಿದ < chunk_ಉಳಿದ ? ಅಡ್ಡಸಾಲು_ಉಳಿದ : chunk_ಉಳಿದ);

            vec_copy_f32(dst_ದತ್ತಾಂಶ + pos, src_ಅಡ್ಡಸಾಲು + ಲಂಬಸಾಲು, n);

            pos += n;
            ಲಂಬಸಾಲು = 0;  // subsequent rows start at column 0
            ಅಡ್ಡಸಾಲು_idx++;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
