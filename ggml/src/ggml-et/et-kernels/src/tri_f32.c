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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

ସଂରଚନା ggml_et_tri_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
    int32_t            tri_type;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    ବାଛ (tri_type) {
        ସ୍ଥିତି TRI_TYPE_LOWER:
            ଫେରନ୍ତୁ i0 < i1;
        ସ୍ଥିତି TRI_TYPE_LOWER_DIAG:
            ଫେରନ୍ତୁ i0 <= i1;
        ସ୍ଥିତି TRI_TYPE_UPPER:
            ଫେରନ୍ତୁ i0 > i1;
        ସ୍ଥିତି TRI_TYPE_UPPER_DIAG:
            ଫେରନ୍ତୁ i0 >= i1;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ 0;
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_tri_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0     = &params->src0;
    ସଂରଚନା ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t ne0 = dst->ne[0];
    ଅଚଳ int64_t ne1 = dst->ne[1];
    ଅଚଳ int64_t ne2 = dst->ne[2];
    ଅଚଳ int64_t ne3 = dst->ne[3];

    ଅଚଳ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ଅଚଳ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ଅଚଳ int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    ଯଦି (ne0 % 16 == 0) {
        ଦଶମିକ zero = 0.0f;
        __asm__ ଅସ୍ଥିର("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ପାଇଁ (int64_t row = thread_id; row < total_rows; row += ଅଙ୍କ_threads) {
            ଅଚଳ int64_t i1 = row % ne1;
            ଅଚଳ int64_t i2 = (row / ne1) % ne2;
            ଅଚଳ int64_t i3 = row / (ne1 * ne2);

            ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ଦଶମିକ *       dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            ବାଛ (tri_type) {
                ସ୍ଥିତି TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ -1;
            }
            ଯଦି (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            ପାଇଁ (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ ଅସ୍ଥିର("fsw.ps f10, %[d]\n" : [d] "=m"(*(ଦଶମିକ (*)[8]) & dst_row[i0])::"f10");
            }
            ପାଇଁ (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            ପାଇଁ (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ ଅସ୍ଥିର(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(ଦଶମିକ (*)[8]) & dst_row[i0])
                    : [s] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src_row[i0])
                    : "f11");
            }
            ପାଇଁ (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            ପାଇଁ (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ ଅସ୍ଥିର("fsw.ps f10, %[d]\n" : [d] "=m"(*(ଦଶମିକ (*)[8]) & dst_row[i0])::"f10");
            }
            ପାଇଁ (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        ଫେରନ୍ତୁ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ଅଚଳ int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        ଅଚଳ int64_t elems_per_cl   = 16;
        ଅଚଳ int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ଅଚଳ int64_t cl_per_thread = (total_cl + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
        ଅଚଳ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        ଯଦି (cl_end > total_cl) {
            cl_end = total_cl;
        }
        ଯଦି (cl_start >= total_cl) {
            ଫେରନ୍ତୁ 0;
        }

        ଅଚଳ int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        ଯଦି (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        ଯେତେବେଳେ (pos < ee) {
            ଅଚଳ int64_t i1 = row_idx % ne1;
            ଅଚଳ int64_t i2 = (row_idx / ne1) % ne2;
            ଅଚଳ int64_t i3 = row_idx / (ne1 * ne2);

            ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            ବାଛ (tri_type) {
                ସ୍ଥିତି TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ -1;
            }
            ଯଦି (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            ପାଇଁ (int64_t i0 = col; i0 < end_col; i0++) {
                ଯଦି (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } ଅନ୍ୟଥା {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    ଫେରନ୍ତୁ 0;
}
