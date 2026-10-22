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

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdbool.h>
#شامل <stdint.h>

ساخت ggml_et_cont_params {
    ساخت ggml_tensor src0;  // F32 input tensor (non-contiguous)
    ساخت ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
ساکن اندرونی خالی vec_copy_f32(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_انتہا = (n / 8) * 8;
    برائے (; i < vec_انتہا; i += 8) {
        __asm__ متلون(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [s] "m"(*(ثابت اعشاری (*)[8]) & src[i])
            : "f10");
    }
    برائے (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
ساکن اندرونی خالی scalar_copy_f32(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

صحیح اندراج_نقطہ(ساخت ggml_et_cont_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت int64_t nb00 = src0->nb[0];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t کل_elements = ne00 * ne01 * ne02 * ne03;

    اگر (کل_elements == 0) {
        واپس 0;
    }

    ثابت bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    اگر (src_contiguous) {
        ثابت int64_t elems_فی_cl = 16;
        ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

        ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
        ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
        int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
        اگر (cl_انتہا > کل_cl) {
            cl_انتہا = کل_cl;
        }
        اگر (cl_آغاز >= کل_cl) {
            واپس 0;
        }

        ثابت int64_t es = cl_آغاز * elems_فی_cl;
        int64_t       ee = cl_انتہا * elems_فی_cl;
        اگر (ee > کل_elements) {
            ee = کل_elements;
        }

        vec_copy_f32(dst_ڈیٹا + es, src0_ڈیٹا + es, (int32_t) (ee - es));
        واپس 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    اگر (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        ثابت int64_t elems_فی_cl = 16;
        ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

        ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
        ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
        int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
        اگر (cl_انتہا > کل_cl) {
            cl_انتہا = کل_cl;
        }
        اگر (cl_آغاز >= کل_cl) {
            واپس 0;
        }

        ثابت int64_t es = cl_آغاز * elems_فی_cl;
        int64_t       ee = cl_انتہا * elems_فی_cl;
        اگر (ee > کل_elements) {
            ee = کل_elements;
        }

        برائے (int64_t idx = es; idx < ee; idx++) {
            ثابت int64_t i00  = idx % ne00;
            ثابت int64_t rem1 = idx / ne00;
            ثابت int64_t i01  = rem1 % ne01;
            ثابت int64_t rem2 = rem1 / ne01;
            ثابت int64_t i02  = rem2 % ne02;
            ثابت int64_t i03  = rem2 / ne02;

            ثابت اعشاری * sp =
                (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_ڈیٹا[idx] = *sp;
        }
        واپس 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    اگر (ne00 % 16 == 0) {
        ثابت int64_t کل_صفیں      = ne01 * ne02 * ne03;
        ثابت int64_t صفیں_فی_thread = (کل_صفیں + عدد_دھاگے - 1) / عدد_دھاگے;
        ثابت int64_t آغاز_صف       = thread_id * صفیں_فی_thread;
        ثابت int64_t انتہا_صف = (آغاز_صف + صفیں_فی_thread < کل_صفیں) ? (آغاز_صف + صفیں_فی_thread) : کل_صفیں;

        اگر (آغاز_صف >= کل_صفیں) {
            واپس 0;
        }

        برائے (int64_t ir = آغاز_صف; ir < انتہا_صف; ir++) {
            ثابت int64_t i03 = ir / (ne02 * ne01);
            ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03);
            اعشاری *       dst_صف = dst_ڈیٹا + ir * ne00;

            vec_copy_f32(dst_صف, src_صف, (int32_t) ne00);
        }
        واپس 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        ثابت int64_t elems_فی_cl = 16;
        ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

        ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
        ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
        int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
        اگر (cl_انتہا > کل_cl) {
            cl_انتہا = کل_cl;
        }
        اگر (cl_آغاز >= کل_cl) {
            واپس 0;
        }

        ثابت int64_t es = cl_آغاز * elems_فی_cl;
        int64_t       ee = cl_انتہا * elems_فی_cl;
        اگر (ee > کل_elements) {
            ee = کل_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t صف_idx = pos / ne00;
        int64_t ستون     = pos % ne00;

        جب (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            ثابت int64_t i03 = صف_idx / (ne02 * ne01);
            ثابت int64_t i02 = (صف_idx - i03 * ne02 * ne01) / ne01;
            ثابت int64_t i01 = صف_idx - i03 * ne02 * ne01 - i02 * ne01;

            ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t صف_بقایا   = ne00 - ستون;
            int64_t chunk_بقایا = ee - pos;
            int32_t n               = (int32_t) (صف_بقایا < chunk_بقایا ? صف_بقایا : chunk_بقایا);

            vec_copy_f32(dst_ڈیٹا + pos, src_صف + ستون, n);

            pos += n;
            ستون = 0;  // subsequent rows start at column 0
            صف_idx++;
        }
    }

    واپس 0;
}
