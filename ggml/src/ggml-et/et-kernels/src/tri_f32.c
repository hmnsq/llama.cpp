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

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

بنية ggml_et_tri_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            tri_type;
};

ساكن مضمن صحيح keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    بدل (tri_type) {
        حالة TRI_TYPE_LOWER:
            ܦܢܐ i0 < i1;
        حالة TRI_TYPE_LOWER_DIAG:
            ܦܢܐ i0 <= i1;
        حالة TRI_TYPE_UPPER:
            ܦܢܐ i0 > i1;
        حالة TRI_TYPE_UPPER_DIAG:
            ܦܢܐ i0 >= i1;
        افتراضي:
            ܦܢܐ 0;
    }
}

صحيح entry_point(بنية ggml_et_tri_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ܐܢ (!kernel_env) {
        ܦܢܐ -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    ܐܢ (thread_id < 0) {
        ܦܢܐ 0;
    }

    ܐܢ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ܦܢܐ -1;
    }

    بنية ggml_tensor * src0     = &params->src0;
    بنية ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    ܐܢ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ܦܢܐ -1;
    }

    عائم * src0_data = (عائم *) src0->data;
    عائم * dst_data  = (عائم *) dst->data;

    ܐܢ (!src0_data || !dst_data) {
        ܦܢܐ -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ثابت int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    ܐܢ (ne0 % 16 == 0) {
        عائم zero = 0.0f;
        __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        لكل (int64_t row = thread_id; row < total_rows; row += num_threads) {
            ثابت int64_t i1 = row % ne1;
            ثابت int64_t i2 = (row / ne1) % ne2;
            ثابت int64_t i3 = row / (ne1 * ne2);

            ثابت عائم * src_row = (ثابت عائم *) ((ثابت حرف *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            عائم *       dst_row = (عائم *) ((حرف *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            بدل (tri_type) {
                حالة TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    اكسر;
                حالة TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    اكسر;
                حالة TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    اكسر;
                افتراضي:
                    ܦܢܐ -1;
            }
            ܐܢ (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            لكل (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_row[i0])::"f10");
            }
            لكل (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            لكل (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ متطاير(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(عائم (*)[8]) & dst_row[i0])
                    : [s] "m"(*(ثابت عائم (*)[8]) & src_row[i0])
                    : "f11");
            }
            لكل (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            لكل (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_row[i0])::"f10");
            }
            لكل (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        ܦܢܐ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ثابت int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elems_per_cl   = 16;
        ثابت int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ثابت int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
        ثابت int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        ܐܢ (cl_end > total_cl) {
            cl_end = total_cl;
        }
        ܐܢ (cl_start >= total_cl) {
            ܦܢܐ 0;
        }

        ثابت int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        ܐܢ (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        ܟܡܐ (pos < ee) {
            ثابت int64_t i1 = row_idx % ne1;
            ثابت int64_t i2 = (row_idx / ne1) % ne2;
            ثابت int64_t i3 = row_idx / (ne1 * ne2);

            ثابت عائم * src_row = (ثابت عائم *) ((ثابت حرف *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            بدل (tri_type) {
                حالة TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    اكسر;
                حالة TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    اكسر;
                حالة TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    اكسر;
                حالة TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    اكسر;
                افتراضي:
                    ܦܢܐ -1;
            }
            ܐܢ (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            لكل (int64_t i0 = col; i0 < end_col; i0++) {
                ܐܢ (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } ܐܚܪܢܐ {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    ܦܢܐ 0;
}
