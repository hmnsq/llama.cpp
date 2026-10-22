//******************************************************************************
// Tri F32 Kernel
// Triangular masking: zero out elements outside the triangular region.
//
// tri_type (matches ggml_tri_type enum):
//   0 = UPPER_DIAG:  keep where i0 >= i1
//   1 = UPPER:       keep where i0 >  i1
//   2 = LOWER_DIAG:  keep where i0 <= i1
//   3 = LOWER:       keep where i0 <  i1
//
// Distribution: cache-line aligned chunks of the flat contiguous dst.
// Each element is individually classified as keep or zero based on its
// (i0, i1) coordinates. This avoids cache-line sharing between threads
// when ne0 is not a multiple of 16.
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

#تعریف_کرو TRI_TYPE_بالائی_DIAG 0
#تعریف_کرو TRI_TYPE_بالائی      1
#تعریف_کرو TRI_TYPE_زیریں_DIAG 2
#تعریف_کرو TRI_TYPE_زیریں      3

ساخت ggml_et_tri_params {
    ساخت ggml_tensor src0;
    ساخت ggml_tensor dst;
    int32_t            tri_type;
};

ساکن اندرونی صحیح keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    انتخاب (tri_type) {
        حالت TRI_TYPE_زیریں:
            واپس i0 < i1;
        حالت TRI_TYPE_زیریں_DIAG:
            واپس i0 <= i1;
        حالت TRI_TYPE_بالائی:
            واپس i0 > i1;
        حالت TRI_TYPE_بالائی_DIAG:
            واپس i0 >= i1;
        طےشدہ:
            واپس 0;
    }
}

صحیح اندراج_نقطہ(ساخت ggml_et_tri_params * params, خالی * env) {
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

    ساخت ggml_tensor * src0     = &params->src0;
    ساخت ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ثابت int64_t کل_صفیں = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    اگر (ne0 % 16 == 0) {
        اعشاری zero = 0.0f;
        __asm__ متلون("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        برائے (int64_t صف = thread_id; صف < کل_صفیں; صف += عدد_دھاگے) {
            ثابت int64_t i1 = صف % ne1;
            ثابت int64_t i2 = (صف / ne1) % ne2;
            ثابت int64_t i3 = صف / (ne1 * ne2);

            ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03);
            اعشاری *       dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_آغاز, keep_انتہا;
            انتخاب (tri_type) {
                حالت TRI_TYPE_زیریں:
                    keep_آغاز = 0;
                    keep_انتہا   = i1;
                    توڑو;
                حالت TRI_TYPE_زیریں_DIAG:
                    keep_آغاز = 0;
                    keep_انتہا   = i1 + 1;
                    توڑو;
                حالت TRI_TYPE_بالائی:
                    keep_آغاز = i1 + 1;
                    keep_انتہا   = ne0;
                    توڑو;
                حالت TRI_TYPE_بالائی_DIAG:
                    keep_آغاز = i1;
                    keep_انتہا   = ne0;
                    توڑو;
                طےشدہ:
                    واپس -1;
            }
            اگر (keep_انتہا > ne0) {
                keep_انتہا = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            برائے (; i0 + 8 <= keep_آغاز; i0 += 8) {
                __asm__ متلون("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_صف[i0])::"f10");
            }
            برائے (; i0 < keep_آغاز; i0++) {
                dst_صف[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            برائے (; i0 + 8 <= keep_انتہا; i0 += 8) {
                __asm__ متلون(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(اعشاری (*)[8]) & dst_صف[i0])
                    : [s] "m"(*(ثابت اعشاری (*)[8]) & src_صف[i0])
                    : "f11");
            }
            برائے (; i0 < keep_انتہا; i0++) {
                dst_صف[i0] = src_صف[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            برائے (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ متلون("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_صف[i0])::"f10");
            }
            برائے (; i0 < ne0; i0++) {
                dst_صف[i0] = 0.0f;
            }
        }
        واپس 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ثابت int64_t کل_elements = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elems_فی_cl   = 16;
        ثابت int64_t کل_cl       = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

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

        int64_t صف_idx = es / ne0;
        int64_t ستون     = es % ne0;

        int64_t pos = es;
        جب (pos < ee) {
            ثابت int64_t i1 = صف_idx % ne1;
            ثابت int64_t i2 = (صف_idx / ne1) % ne2;
            ثابت int64_t i3 = صف_idx / (ne1 * ne2);

            ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t صف_بقایا   = ne0 - ستون;
            int64_t chunk_بقایا = ee - pos;
            int64_t n               = صف_بقایا < chunk_بقایا ? صف_بقایا : chunk_بقایا;

            int64_t keep_آغاز, keep_انتہا;
            انتخاب (tri_type) {
                حالت TRI_TYPE_زیریں:
                    keep_آغاز = 0;
                    keep_انتہا   = i1;
                    توڑو;
                حالت TRI_TYPE_زیریں_DIAG:
                    keep_آغاز = 0;
                    keep_انتہا   = i1 + 1;
                    توڑو;
                حالت TRI_TYPE_بالائی:
                    keep_آغاز = i1 + 1;
                    keep_انتہا   = ne0;
                    توڑو;
                حالت TRI_TYPE_بالائی_DIAG:
                    keep_آغاز = i1;
                    keep_انتہا   = ne0;
                    توڑو;
                طےشدہ:
                    واپس -1;
            }
            اگر (keep_انتہا > ne0) {
                keep_انتہا = ne0;
            }

            int64_t انتہا_ستون = ستون + n;
            برائے (int64_t i0 = ستون; i0 < انتہا_ستون; i0++) {
                اگر (i0 >= keep_آغاز && i0 < keep_انتہا) {
                    dst_ڈیٹا[pos + (i0 - ستون)] = src_صف[i0];
                } ورنہ {
                    dst_ڈیٹا[pos + (i0 - ستون)] = 0.0f;
                }
            }

            pos += n;
            ستون = 0;
            صف_idx++;
        }
    }

    واپس 0;
}
