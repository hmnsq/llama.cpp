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
    ثابت int32_t vec_پای_ټکی = (n / 8) * 8;
    لكل (; i < vec_پای_ټکی; i += 8) {
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

صحيح ثبت_ټکی(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

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

    ثابت int64_t ټول_elements = ne00 * ne01 * ne02 * ne03;

    إذا (ټول_elements == 0) {
        أرجع 0;
    }

    ثابت bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    إذا (src_contiguous) {
        ثابت int64_t elems_فی_cl = 16;
        ثابت int64_t ټول_cl     = (ټول_elements + elems_فی_cl - 1) / elems_فی_cl;

        ثابت int64_t cl_فی_thread = (ټول_cl + عدد_تارونه - 1) / عدد_تارونه;
        ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
        int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
        إذا (cl_پای_ټکی > ټول_cl) {
            cl_پای_ټکی = ټول_cl;
        }
        إذا (cl_پیل >= ټول_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_پیل * elems_فی_cl;
        int64_t       ee = cl_پای_ټکی * elems_فی_cl;
        إذا (ee > ټول_elements) {
            ee = ټول_elements;
        }

        vec_copy_f32(dst_معلومات + es, src0_معلومات + es, (int32_t) (ee - es));
        أرجع 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    إذا (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        ثابت int64_t elems_فی_cl = 16;
        ثابت int64_t ټول_cl     = (ټول_elements + elems_فی_cl - 1) / elems_فی_cl;

        ثابت int64_t cl_فی_thread = (ټول_cl + عدد_تارونه - 1) / عدد_تارونه;
        ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
        int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
        إذا (cl_پای_ټکی > ټول_cl) {
            cl_پای_ټکی = ټول_cl;
        }
        إذا (cl_پیل >= ټول_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_پیل * elems_فی_cl;
        int64_t       ee = cl_پای_ټکی * elems_فی_cl;
        إذا (ee > ټول_elements) {
            ee = ټول_elements;
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
        ثابت int64_t ټول_لیکې      = ne01 * ne02 * ne03;
        ثابت int64_t لیکې_فی_thread = (ټول_لیکې + عدد_تارونه - 1) / عدد_تارونه;
        ثابت int64_t پیل_لیکه       = thread_id * لیکې_فی_thread;
        ثابت int64_t پای_ټکی_لیکه = (پیل_لیکه + لیکې_فی_thread < ټول_لیکې) ? (پیل_لیکه + لیکې_فی_thread) : ټول_لیکې;

        إذا (پیل_لیکه >= ټول_لیکې) {
            أرجع 0;
        }

        لكل (int64_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ir++) {
            ثابت int64_t i03 = ir / (ne02 * ne01);
            ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);
            عائم *       dst_لیکه = dst_معلومات + ir * ne00;

            vec_copy_f32(dst_لیکه, src_لیکه, (int32_t) ne00);
        }
        أرجع 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        ثابت int64_t elems_فی_cl = 16;
        ثابت int64_t ټول_cl     = (ټول_elements + elems_فی_cl - 1) / elems_فی_cl;

        ثابت int64_t cl_فی_thread = (ټول_cl + عدد_تارونه - 1) / عدد_تارونه;
        ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
        int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
        إذا (cl_پای_ټکی > ټول_cl) {
            cl_پای_ټکی = ټول_cl;
        }
        إذا (cl_پیل >= ټول_cl) {
            أرجع 0;
        }

        ثابت int64_t es = cl_پیل * elems_فی_cl;
        int64_t       ee = cl_پای_ټکی * elems_فی_cl;
        إذا (ee > ټول_elements) {
            ee = ټول_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t لیکه_idx = pos / ne00;
        int64_t ستون     = pos % ne00;

        طالما (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            ثابت int64_t i03 = لیکه_idx / (ne02 * ne01);
            ثابت int64_t i02 = (لیکه_idx - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = لیکه_idx - i03 * ne02 * ne01 - i02 * ne01;

            ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t لیکه_پاتې_شوی   = ne00 - ستون;
            int64_t chunk_پاتې_شوی = ee - pos;
            int32_t n               = (int32_t) (لیکه_پاتې_شوی < chunk_پاتې_شوی ? لیکه_پاتې_شوی : chunk_پاتې_شوی);

            vec_copy_f32(dst_معلومات + pos, src_لیکه + ستون, n);

            pos += n;
            ستون = 0;  // subsequent rows start at column 0
            لیکه_idx++;
        }
    }

    أرجع 0;
}
