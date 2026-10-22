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

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>
#شامل <string.h>

ساخت ggml_et_concat_params {
    ساخت ggml_tensor src0;  // F32 input tensor 0
    ساخت ggml_tensor src1;  // F32 input tensor 1
    ساخت ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ساکن اندرونی خالی copy_صف_aligned(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [src_vec] "m"(*(ثابت اعشاری (*)[8]) & src[i])
            : "f11");
    }
}

صحیح اندراج_نقطہ(ساخت ggml_et_concat_params * params, خالی * env) {
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
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * src1_ڈیٹا = (اعشاری *) src1->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
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
    ثابت int64_t کل_صفیں = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    اگر (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != حجم(اعشاری) || nb10 != حجم(اعشاری))) {
        ثابت int64_t صفیں_فی_گروہ = et_صفیں_فی_cacheline_گروہ(ne0, حجم(اعشاری));
        ثابت int64_t کل_groups   = (کل_صفیں + صفیں_فی_گروہ - 1) / صفیں_فی_گروہ;

        برائے (int64_t grp = thread_id; grp < کل_groups; grp += عدد_دھاگے) {
            ثابت int64_t صف_آغاز = grp * صفیں_فی_گروہ;
            int64_t       صف_انتہا   = صف_آغاز + صفیں_فی_گروہ;
            اگر (صف_انتہا > کل_صفیں) {
                صف_انتہا = کل_صفیں;
            }

            برائے (int64_t صف = صف_آغاز; صف < صف_انتہا; صف++) {
                int64_t i1 = صف % ne1;
                int64_t i2 = (صف / ne1) % ne2;
                int64_t i3 = صف / (ne1 * ne2);

                اعشاری * dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                ثابت حرف * s0_بنیاد = (ثابت حرف *) src0_ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03;
                برائے (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_صف[i0] = *(ثابت اعشاری *) (s0_بنیاد + i0 * nb00);
                }

                ثابت حرف * s1_بنیاد = (ثابت حرف *) src1_ڈیٹا + i1 * nb11 + i2 * nb12 + i3 * nb13;
                برائے (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_صف[ne00 + i0] = *(ثابت اعشاری *) (s1_بنیاد + i0 * nb10);
                }
            }
        }
        واپس 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    برائے (int64_t صف = thread_id; صف < کل_صفیں; صف += عدد_دھاگے) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = صف % ne1;
        int64_t i2 = (صف / ne1) % ne2;
        int64_t i3 = صف / (ne1 * ne2);

        اعشاری * dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        اگر (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            ثابت اعشاری * s0_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ثابت اعشاری * s1_صف = (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne00);
            copy_صف_aligned(dst_صف + ne00, s1_صف, (int32_t) ne10);

        } ورنہ اگر (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            اگر (i1 < ne01) {
                ثابت اعشاری * s0_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne0);
            } ورنہ {
                ثابت اعشاری * s1_صف =
                    (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_صف_aligned(dst_صف, s1_صف, (int32_t) ne0);
            }

        } ورنہ اگر (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            اگر (i2 < ne02) {
                ثابت اعشاری * s0_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne0);
            } ورنہ {
                ثابت اعشاری * s1_صف =
                    (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_صف_aligned(dst_صف, s1_صف, (int32_t) ne0);
            }

        } ورنہ {
            // dim == 3: first ne03 batches from src0, rest from src1
            اگر (i3 < ne03) {
                ثابت اعشاری * s0_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_صف_aligned(dst_صف, s0_صف, (int32_t) ne0);
            } ورنہ {
                ثابت اعشاری * s1_صف =
                    (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_صف_aligned(dst_صف, s1_صف, (int32_t) ne0);
            }
        }
    }

    واپس 0;
}
