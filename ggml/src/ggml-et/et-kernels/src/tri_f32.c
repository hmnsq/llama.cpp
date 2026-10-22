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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

কাঠামো ggml_et_tri_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor dst;
    int32_t            tri_type;
};

স্থির অন্তঃরেখ পূর্ণসংখ্যা keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    নির্বাচন (tri_type) {
        ক্ষেত্র TRI_TYPE_LOWER:
            ফেরত i0 < i1;
        ক্ষেত্র TRI_TYPE_LOWER_DIAG:
            ফেরত i0 <= i1;
        ক্ষেত্র TRI_TYPE_UPPER:
            ফেরত i0 > i1;
        ক্ষেত্র TRI_TYPE_UPPER_DIAG:
            ফেরত i0 >= i1;
        পূর্বনির্ধারিত:
            ফেরত 0;
    }
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_tri_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0     = &params->src0;
    কাঠামো ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t ne0 = dst->ne[0];
    ধ্রুবক int64_t ne1 = dst->ne[1];
    ধ্রুবক int64_t ne2 = dst->ne[2];
    ধ্রুবক int64_t ne3 = dst->ne[3];

    ধ্রুবক size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুবক size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ধ্রুবক int64_t যোগ_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    যদি (ne0 % 16 == 0) {
        দশমিক zero = 0.0f;
        __asm__ অস্থির("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        জন্য (int64_t row = thread_id; row < যোগ_rows; row += অঙ্ক_threads) {
            ধ্রুবক int64_t i1 = row % ne1;
            ধ্রুবক int64_t i2 = (row / ne1) % ne2;
            ধ্রুবক int64_t i3 = row / (ne1 * ne2);

            ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i1 * nb01 + i2 * nb02 + i3 * nb03);
            দশমিক *       dst_row = (দশমিক *) ((অক্ষর *) dst_তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            নির্বাচন (tri_type) {
                ক্ষেত্র TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    বিরতি;
                ক্ষেত্র TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    বিরতি;
                ক্ষেত্র TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    বিরতি;
                ক্ষেত্র TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    বিরতি;
                পূর্বনির্ধারিত:
                    ফেরত -1;
            }
            যদি (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            জন্য (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ অস্থির("fsw.ps f10, %[d]\n" : [d] "=m"(*(দশমিক (*)[8]) & dst_row[i0])::"f10");
            }
            জন্য (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            জন্য (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ অস্থির(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(দশমিক (*)[8]) & dst_row[i0])
                    : [s] "m"(*(ধ্রুবক দশমিক (*)[8]) & src_row[i0])
                    : "f11");
            }
            জন্য (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            জন্য (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ অস্থির("fsw.ps f10, %[d]\n" : [d] "=m"(*(দশমিক (*)[8]) & dst_row[i0])::"f10");
            }
            জন্য (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        ফেরত 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ধ্রুবক int64_t যোগ_elements = ne0 * ne1 * ne2 * ne3;
        ধ্রুবক int64_t elems_per_cl   = 16;
        ধ্রুবক int64_t যোগ_cl       = (যোগ_elements + elems_per_cl - 1) / elems_per_cl;

        ধ্রুবক int64_t cl_per_thread = (যোগ_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
        ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        যদি (cl_end > যোগ_cl) {
            cl_end = যোগ_cl;
        }
        যদি (cl_start >= যোগ_cl) {
            ফেরত 0;
        }

        ধ্রুবক int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        যদি (ee > যোগ_elements) {
            ee = যোগ_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        যখন (pos < ee) {
            ধ্রুবক int64_t i1 = row_idx % ne1;
            ধ্রুবক int64_t i2 = (row_idx / ne1) % ne2;
            ধ্রুবক int64_t i3 = row_idx / (ne1 * ne2);

            ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            নির্বাচন (tri_type) {
                ক্ষেত্র TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    বিরতি;
                ক্ষেত্র TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    বিরতি;
                ক্ষেত্র TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    বিরতি;
                ক্ষেত্র TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    বিরতি;
                পূর্বনির্ধারিত:
                    ফেরত -1;
            }
            যদি (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            জন্য (int64_t i0 = col; i0 < end_col; i0++) {
                যদি (i0 >= keep_start && i0 < keep_end) {
                    dst_তথ্য[pos + (i0 - col)] = src_row[i0];
                } অন্যথা {
                    dst_তথ্য[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    ফেরত 0;
}
