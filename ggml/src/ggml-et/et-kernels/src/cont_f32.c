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

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdbool.h>
#ضمن <stdint.h>

بنية ggml_et_cont_params {
    بنية ggml_tensor src0;  // F32 input tensor (non-contiguous)
    بنية ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
ساكن مضمن فراغ vec_copy_f32(عائم * dst, ثابت عائم * src, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_پڇاڙي = (n / 8) * 8;
    لكل (; i < vec_پڇاڙي; i += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(عائم (*)[8]) & dst[i])
            : [s] "m"(*(ثابت عائم (*)[8]) & src[i])
            : "f10");
    }
    لكل (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
ساكن مضمن فراغ scalar_copy_f32(عائم * dst, ثابت عائم * src, int32_t n) {
    لكل (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

صحيح داخلا_نقطو(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    إذا (!src0_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت int64_t nb00 = src0->nb[0];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t ڪل_elements = ne00 * ne01 * ne02 * ne03;

    إذا (ڪل_elements == 0) {
        أرجع 0;
    }

    ثابت bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    إذا (src_contiguous) {
        ثابت int64_t elems_في_cl = 16;
        ثابت int64_t ڪل_cl     = (ڪل_elements + elems_في_cl - 1) / elems_في_cl;

        ثابت int64_t cl_في_thread = (ڪل_cl + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
        ثابت int64_t cl_شروعات      = thread_id * cl_في_thread;
        int64_t       cl_پڇاڙي        = cl_شروعات + cl_في_thread;
        إذا (cl_پڇاڙي > ڪل_cl) {
            cl_پڇاڙي = ڪل_cl;
        }
        إذا (cl_شروعات >= ڪل_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_شروعات * elems_في_cl;
        int64_t       ee = cl_پڇاڙي * elems_في_cl;
        إذا (ee > ڪل_elements) {
            ee = ڪل_elements;
        }

        vec_copy_f32(dst_معلومات + es, src0_معلومات + es, (int32_t) (ee - es));
        أرجع 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    إذا (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        ثابت int64_t elems_في_cl = 16;
        ثابت int64_t ڪل_cl     = (ڪل_elements + elems_في_cl - 1) / elems_في_cl;

        ثابت int64_t cl_في_thread = (ڪل_cl + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
        ثابت int64_t cl_شروعات      = thread_id * cl_في_thread;
        int64_t       cl_پڇاڙي        = cl_شروعات + cl_في_thread;
        إذا (cl_پڇاڙي > ڪل_cl) {
            cl_پڇاڙي = ڪل_cl;
        }
        إذا (cl_شروعات >= ڪل_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_شروعات * elems_في_cl;
        int64_t       ee = cl_پڇاڙي * elems_في_cl;
        إذا (ee > ڪل_elements) {
            ee = ڪل_elements;
        }

        لكل (int64_t idx = es; idx < ee; idx++) {
            ثابت int64_t i00  = idx % ne00;
            ثابت int64_t rem1 = idx / ne00;
            ثابت int64_t i01  = rem1 % ne01;
            ثابت int64_t rem2 = rem1 / ne01;
            ثابت int64_t i02  = rem2 % ne02;
            ثابت int64_t i03  = rem2 / ne02;

            ثابت عائم * sp =
                (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_معلومات[idx] = *sp;
        }
        أرجع 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    إذا (ne00 % 16 == 0) {
        ثابت int64_t ڪل_صفون      = ne01 * ne02 * ne03;
        ثابت int64_t صفون_في_thread = (ڪل_صفون + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
        ثابت int64_t شروعات_صف       = thread_id * صفون_في_thread;
        ثابت int64_t پڇاڙي_صف = (شروعات_صف + صفون_في_thread < ڪل_صفون) ? (شروعات_صف + صفون_في_thread) : ڪل_صفون;

        إذا (شروعات_صف >= ڪل_صفون) {
            أرجع 0;
        }

        لكل (int64_t ir = شروعات_صف; ir < پڇاڙي_صف; ir++) {
            ثابت int64_t i03 = ir / (ne02 * ne01);
            ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);
            عائم *       dst_صف = dst_معلومات + ir * ne00;

            vec_copy_f32(dst_صف, src_صف, (int32_t) ne00);
        }
        أرجع 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        ثابت int64_t elems_في_cl = 16;
        ثابت int64_t ڪل_cl     = (ڪل_elements + elems_في_cl - 1) / elems_في_cl;

        ثابت int64_t cl_في_thread = (ڪل_cl + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
        ثابت int64_t cl_شروعات      = thread_id * cl_في_thread;
        int64_t       cl_پڇاڙي        = cl_شروعات + cl_في_thread;
        إذا (cl_پڇاڙي > ڪل_cl) {
            cl_پڇاڙي = ڪل_cl;
        }
        إذا (cl_شروعات >= ڪل_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_شروعات * elems_في_cl;
        int64_t       ee = cl_پڇاڙي * elems_في_cl;
        إذا (ee > ڪل_elements) {
            ee = ڪل_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t صف_idx = pos / ne00;
        int64_t کانو     = pos % ne00;

        طالما (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            ثابت int64_t i03 = صف_idx / (ne02 * ne01);
            ثابت int64_t i02 = (صف_idx - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = صف_idx - i03 * ne02 * ne01 - i02 * ne01;

            ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t صف_بچيل   = ne00 - کانو;
            int64_t chunk_بچيل = ee - pos;
            int32_t n               = (int32_t) (صف_بچيل < chunk_بچيل ? صف_بچيل : chunk_بچيل);

            vec_copy_f32(dst_معلومات + pos, src_صف + کانو, n);

            pos += n;
            کانو = 0;  // subsequent rows start at column 0
            صف_idx++;
        }
    }

    أرجع 0;
}
