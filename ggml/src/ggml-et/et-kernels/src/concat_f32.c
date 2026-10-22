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

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>
#ضمن <string.h>

بنية ggml_et_concat_params {
    بنية ggml_tensor src0;  // F32 input tensor 0
    بنية ggml_tensor src1;  // F32 input tensor 1
    بنية ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ساكن مضمن فراغ copy_صف_aligned(عائم * dst, ثابت عائم * src, int32_t n) {
    لكل (int32_t i = 0; i < n; i += 8) {
        __asm__ متطاير(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(عائم (*)[8]) & dst[i])
            : [src_vec] "m"(*(ثابت عائم (*)[8]) & src[i])
            : "f11");
    }
}

صحيح داخلا_نقطو(بنية ggml_et_concat_params * params, فراغ * env) {
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
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * src1_معلومات = (عائم *) src1->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !src1_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ثابت int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    ثابت int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    ثابت size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    ثابت size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    ثابت int64_t ڪل_صفون = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    إذا (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != حجم(عائم) || nb10 != حجم(عائم))) {
        ثابت int64_t صفون_في_گروهه = et_صفون_في_cacheline_گروهه(ne0, حجم(عائم));
        ثابت int64_t ڪل_groups   = (ڪل_صفون + صفون_في_گروهه - 1) / صفون_في_گروهه;

        لكل (int64_t grp = thread_id; grp < ڪل_groups; grp += انگ_ڌاڳا) {
            ثابت int64_t صف_شروعات = grp * صفون_في_گروهه;
            int64_t       صف_پڇاڙي   = صف_شروعات + صفون_في_گروهه;
            إذا (صف_پڇاڙي > ڪل_صفون) {
                صف_پڇاڙي = ڪل_صفون;
            }

            لكل (int64_t صف = صف_شروعات; صف < صف_پڇاڙي; صف++) {
                int64_t i1 = صف % ne1;
                int64_t i2 = (صف / ne1) % ne2;
                int64_t i3 = صف / (ne1 * ne2);

                عائم * dst_صف = (عائم *) ((حرف *) dst_معلومات + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                ثابت حرف * s0_بنياد = (ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03;
                لكل (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_صف[i0] = *(ثابت عائم *) (s0_بنياد + i0 * nb00);
                }

                ثابت حرف * s1_بنياد = (ثابت حرف *) src1_معلومات + i1 * nb11 + i2 * nb12 + i3 * nb13;
                لكل (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_صف[ne00 + i0] = *(ثابت عائم *) (s1_بنياد + i0 * nb10);
                }
            }
        }
        أرجع 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    لكل (int64_t صف = thread_id; صف < ڪل_صفون; صف += انگ_ڌاڳا) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = صف % ne1;
        int64_t i2 = (صف / ne1) % ne2;
        int64_t i3 = صف / (ne1 * ne2);

        عائم * dst_صف = (عائم *) ((حرف *) dst_معلومات + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        إذا (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            ثابت عائم * s0_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ثابت عائم * s1_صف = (ثابت عائم *) ((ثابت حرف *) src1_معلومات + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne00);
            copy_صف_aligned(dst_صف + ne00, s1_صف, (int32_t) ne10);

        } وإلا إذا (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            إذا (i1 < ne01) {
                ثابت عائم * s0_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne0);
            } وإلا {
                ثابت عائم * s1_صف =
                    (ثابت عائم *) ((ثابت حرف *) src1_معلومات + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_صف_aligned(dst_صف, s1_صف, (int32_t) ne0);
            }

        } وإلا إذا (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            إذا (i2 < ne02) {
                ثابت عائم * s0_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne0);
            } وإلا {
                ثابت عائم * s1_صف =
                    (ثابت عائم *) ((ثابت حرف *) src1_معلومات + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_صف_aligned(dst_صف, s1_صف, (int32_t) ne0);
            }

        } وإلا {
            // dim == 3: first ne03 batches from src0, rest from src1
            إذا (i3 < ne03) {
                ثابت عائم * s0_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne0);
            } وإلا {
                ثابت عائم * s1_صف =
                    (ثابت عائم *) ((ثابت حرف *) src1_معلومات + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_صف_aligned(dst_صف, s1_صف, (int32_t) ne0);
            }
        }
    }

    أرجع 0;
}
