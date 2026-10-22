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

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

ساختار ggml_et_tri_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor dst;
    int32_t            tri_type;
};

ایستا درونخطی صحیح keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    انتخاب (tri_type) {
        حالت TRI_TYPE_LOWER:
            بازگشت i0 < i1;
        حالت TRI_TYPE_LOWER_DIAG:
            بازگشت i0 <= i1;
        حالت TRI_TYPE_UPPER:
            بازگشت i0 > i1;
        حالت TRI_TYPE_UPPER_DIAG:
            بازگشت i0 >= i1;
        پیشفرض:
            بازگشت 0;
    }
}

صحیح entry_point(ساختار ggml_et_tri_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0     = &params->src0;
    ساختار ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    اعشاری * src0_data = (اعشاری *) src0->data;
    اعشاری * dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;
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
    اگر (ne0 % 16 == 0) {
        اعشاری zero = 0.0f;
        __asm__ فرار("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        برای (int64_t row = thread_id; row < total_rows; row += رقم_threads) {
            ثابت int64_t i1 = row % ne1;
            ثابت int64_t i2 = (row / ne1) % ne2;
            ثابت int64_t i3 = row / (ne1 * ne2);

            ثابت اعشاری * src_row = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            اعشاری *       dst_row = (اعشاری *) ((نویسه *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            انتخاب (tri_type) {
                حالت TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    بشکن;
                حالت TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    بشکن;
                حالت TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    بشکن;
                حالت TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    بشکن;
                پیشفرض:
                    بازگشت -1;
            }
            اگر (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            برای (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ فرار("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_row[i0])::"f10");
            }
            برای (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            برای (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ فرار(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(اعشاری (*)[8]) & dst_row[i0])
                    : [s] "m"(*(ثابت اعشاری (*)[8]) & src_row[i0])
                    : "f11");
            }
            برای (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            برای (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ فرار("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_row[i0])::"f10");
            }
            برای (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        بازگشت 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ثابت int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elems_per_cl   = 16;
        ثابت int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ثابت int64_t cl_per_thread = (total_cl + رقم_threads - 1) / رقم_threads;
        ثابت int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        اگر (cl_end > total_cl) {
            cl_end = total_cl;
        }
        اگر (cl_start >= total_cl) {
            بازگشت 0;
        }

        ثابت int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        اگر (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        وقتی (pos < ee) {
            ثابت int64_t i1 = row_idx % ne1;
            ثابت int64_t i2 = (row_idx / ne1) % ne2;
            ثابت int64_t i3 = row_idx / (ne1 * ne2);

            ثابت اعشاری * src_row = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            انتخاب (tri_type) {
                حالت TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    بشکن;
                حالت TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    بشکن;
                حالت TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    بشکن;
                حالت TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    بشکن;
                پیشفرض:
                    بازگشت -1;
            }
            اگر (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            برای (int64_t i0 = col; i0 < end_col; i0++) {
                اگر (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } وگرنه {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    بازگشت 0;
}
