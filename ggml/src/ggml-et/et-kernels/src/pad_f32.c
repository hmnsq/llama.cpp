//******************************************************************************
// Bare Metal PAD F32 Kernel
// Zero-pads an F32 tensor along dimensions 1-3.
//
// Constraints:
//   - No dim0 padding (lp[0]==0, rp[0]==0)
//   - dst contiguous
//   - src nb[0] == 4 (dim0 contiguous for vectorized reads)
//   - Zero-pad only (no circular mode)
//
// Two paths:
//   Aligned (ne0 % 16 == 0): rows distributed across harts, vectorized.
//   Small   (16 % ne0 == 0): cache-line distributed, scalar per-element.
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_pad_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
स्थिर ओळीत शून्य vec_copy_f32(दशांश * dst, स्थिरांक दशांश * src, int32_t n) {
    int32_t       i       = 0;
    स्थिरांक int32_t vec_end = (n / 8) * 8;
    साठी (; i < vec_end; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(दशांश (*)[8]) & dst[i])
            : [s] "m"(*(स्थिरांक दशांश (*)[8]) & src[i])
            : "f10");
    }
    साठी (; i < n; i++) {
        dst[i] = src[i];
    }
}

पूर्णांक entry_point(संरचना ggml_et_pad_params * params, शून्य * env) {
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

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक दशांश * src0_data = (स्थिरांक दशांश *) src0->data;
    दशांश *       dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !dst_data) {
        परत -1;
    }

    // Dst dimensions
    स्थिरांक int64_t ne0 = dst->ne[0];
    स्थिरांक int64_t ne1 = dst->ne[1];
    स्थिरांक int64_t ne2 = dst->ne[2];
    स्थिरांक int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    स्थिरांक int64_t nb1_src = src0->nb[1];
    स्थिरांक int64_t nb2_src = src0->nb[2];
    स्थिरांक int64_t nb3_src = src0->nb[3];

    // Padding values
    स्थिरांक int32_t lp1 = params->lp[1];
    स्थिरांक int32_t rp1 = params->rp[1];
    स्थिरांक int32_t lp2 = params->lp[2];
    स्थिरांक int32_t rp2 = params->rp[2];
    स्थिरांक int32_t lp3 = params->lp[3];
    स्थिरांक int32_t rp3 = params->rp[3];

    स्थिरांक int64_t total_rows     = ne1 * ne2 * ne3;
    स्थिरांक int64_t total_elements = ne0 * total_rows;

    जर (total_elements == 0) {
        परत 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    दशांश zero = 0.0f;
    __asm__ अस्थिर("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    जर (ne0 % 16 == 0) {
        साठी (int64_t row = thread_id; row < total_rows; row += अंक_threads) {
            स्थिरांक int64_t i3 = row / (ne1 * ne2);
            स्थिरांक int64_t i2 = (row / ne1) % ne2;
            स्थिरांक int64_t i1 = row % ne1;

            दशांश * dst_row = dst_data + row * ne0;

            जर (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                स्थिरांक दशांश * src_row = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } अन्यथा {
                int64_t       i       = 0;
                स्थिरांक int64_t vec_end = (ne0 / 8) * 8;
                साठी (; i < vec_end; i += 8) {
                    __asm__ अस्थिर("fsw.ps f12, %[d]\n" : [d] "=m"(*(दशांश (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        परत 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    स्थिरांक int64_t elems_per_cl = 16;
    स्थिरांक int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    स्थिरांक int64_t ne1_data_end = ne1 - rp1;
    स्थिरांक int64_t ne2_data_end = ne2 - rp2;
    स्थिरांक int64_t ne3_data_end = ne3 - rp3;

    साठी (int64_t cl = thread_id; cl < total_cl; cl += अंक_threads) {
        स्थिरांक int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        जर (elem_end > total_elements) {
            elem_end = total_elements;
        }

        साठी (int64_t idx = elem_start; idx < elem_end; idx++) {
            स्थिरांक int64_t i0   = idx % ne0;
            स्थिरांक int64_t rem  = idx / ne0;
            स्थिरांक int64_t i1   = rem % ne1;
            स्थिरांक int64_t rem2 = rem / ne1;
            स्थिरांक int64_t i2   = rem2 % ne2;
            स्थिरांक int64_t i3   = rem2 / ne2;

            जर (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                स्थिरांक दशांश * sp = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } अन्यथा {
                dst_data[idx] = 0.0f;
            }
        }
    }

    परत 0;
}
