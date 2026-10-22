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

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

সংরচনা ggml_et_tri_params {
    সংরচনা ggml_tensor src0;
    সংরচনা ggml_tensor dst;
    int32_t            tri_type;
};

স্থির অন্তঃস্থ পূর্ণাঙ্ক keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    ৱিকল্প (tri_type) {
        পক্ষ TRI_TYPE_LOWER:
            ঘূৰাই i0 < i1;
        পক্ষ TRI_TYPE_LOWER_DIAG:
            ঘূৰাই i0 <= i1;
        পক্ষ TRI_TYPE_UPPER:
            ঘূৰাই i0 > i1;
        পক্ষ TRI_TYPE_UPPER_DIAG:
            ঘূৰাই i0 >= i1;
        স্ৱভাৱ:
            ঘূৰাই 0;
    }
}

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_tri_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0     = &params->src0;
    সংরচনা ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t ne0 = dst->ne[0];
    ধ্রুৱ int64_t ne1 = dst->ne[1];
    ধ্রুৱ int64_t ne2 = dst->ne[2];
    ধ্রুৱ int64_t ne3 = dst->ne[3];

    ধ্রুৱ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুৱ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ধ্রুৱ int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    যদি (ne0 % 16 == 0) {
        ভিন্ন zero = 0.0f;
        __asm__ চঞ্চল("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        বাবে (int64_t row = thread_id; row < total_rows; row += অঙ্ক_threads) {
            ধ্রুৱ int64_t i1 = row % ne1;
            ধ্রুৱ int64_t i2 = (row / ne1) % ne2;
            ধ্রুৱ int64_t i3 = row / (ne1 * ne2);

            ধ্রুৱ ভিন্ন * src_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ভিন্ন *       dst_row = (ভিন্ন *) ((ৱর্ণ *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            ৱিকল্প (tri_type) {
                পক্ষ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    ভঙ্গ;
                পক্ষ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    ভঙ্গ;
                পক্ষ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    ভঙ্গ;
                পক্ষ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    ভঙ্গ;
                স্ৱভাৱ:
                    ঘূৰাই -1;
            }
            যদি (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            বাবে (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ চঞ্চল("fsw.ps f10, %[d]\n" : [d] "=m"(*(ভিন্ন (*)[8]) & dst_row[i0])::"f10");
            }
            বাবে (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            বাবে (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ চঞ্চল(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(ভিন্ন (*)[8]) & dst_row[i0])
                    : [s] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src_row[i0])
                    : "f11");
            }
            বাবে (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            বাবে (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ চঞ্চল("fsw.ps f10, %[d]\n" : [d] "=m"(*(ভিন্ন (*)[8]) & dst_row[i0])::"f10");
            }
            বাবে (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        ঘূৰাই 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ধ্রুৱ int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        ধ্রুৱ int64_t elems_per_cl   = 16;
        ধ্রুৱ int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ধ্রুৱ int64_t cl_per_thread = (total_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
        ধ্রুৱ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        যদি (cl_end > total_cl) {
            cl_end = total_cl;
        }
        যদি (cl_start >= total_cl) {
            ঘূৰাই 0;
        }

        ধ্রুৱ int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        যদি (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        যেতিয়ালৈ (pos < ee) {
            ধ্রুৱ int64_t i1 = row_idx % ne1;
            ধ্রুৱ int64_t i2 = (row_idx / ne1) % ne2;
            ধ্রুৱ int64_t i3 = row_idx / (ne1 * ne2);

            ধ্রুৱ ভিন্ন * src_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            ৱিকল্প (tri_type) {
                পক্ষ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    ভঙ্গ;
                পক্ষ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    ভঙ্গ;
                পক্ষ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    ভঙ্গ;
                পক্ষ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    ভঙ্গ;
                স্ৱভাৱ:
                    ঘূৰাই -1;
            }
            যদি (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            বাবে (int64_t i0 = col; i0 < end_col; i0++) {
                যদি (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } অন্যথা {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    ঘূৰাই 0;
}
