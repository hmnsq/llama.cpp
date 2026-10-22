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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

கட்டமைப்பு ggml_et_tri_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
    int32_t            tri_type;
};

நிலையான உள்வரி முழு keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    தேர்வு (tri_type) {
        நிலை TRI_TYPE_LOWER:
            திரும்பு i0 < i1;
        நிலை TRI_TYPE_LOWER_DIAG:
            திரும்பு i0 <= i1;
        நிலை TRI_TYPE_UPPER:
            திரும்பு i0 > i1;
        நிலை TRI_TYPE_UPPER_DIAG:
            திரும்பு i0 >= i1;
        இயல்பு:
            திரும்பு 0;
    }
}

முழு entry_point(கட்டமைப்பு ggml_et_tri_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0     = &params->src0;
    கட்டமைப்பு ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t ne0 = dst->ne[0];
    மாறிலி int64_t ne1 = dst->ne[1];
    மாறிலி int64_t ne2 = dst->ne[2];
    மாறிலி int64_t ne3 = dst->ne[3];

    மாறிலி size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    மாறிலி size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    மாறிலி int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    எனில் (ne0 % 16 == 0) {
        தசம zero = 0.0f;
        __asm__ நிலையற்ற("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        க்கு (int64_t row = thread_id; row < total_rows; row += இலக்கம்_threads) {
            மாறிலி int64_t i1 = row % ne1;
            மாறிலி int64_t i2 = (row / ne1) % ne2;
            மாறிலி int64_t i3 = row / (ne1 * ne2);

            மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            தசம *       dst_row = (தசம *) ((எழுத்து *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            தேர்வு (tri_type) {
                நிலை TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    நிறுத்து;
                நிலை TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    நிறுத்து;
                நிலை TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    நிறுத்து;
                நிலை TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    நிறுத்து;
                இயல்பு:
                    திரும்பு -1;
            }
            எனில் (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            க்கு (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ நிலையற்ற("fsw.ps f10, %[d]\n" : [d] "=m"(*(தசம (*)[8]) & dst_row[i0])::"f10");
            }
            க்கு (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            க்கு (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ நிலையற்ற(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(தசம (*)[8]) & dst_row[i0])
                    : [s] "m"(*(மாறிலி தசம (*)[8]) & src_row[i0])
                    : "f11");
            }
            க்கு (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            க்கு (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ நிலையற்ற("fsw.ps f10, %[d]\n" : [d] "=m"(*(தசம (*)[8]) & dst_row[i0])::"f10");
            }
            க்கு (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        திரும்பு 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        மாறிலி int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        மாறிலி int64_t elems_per_cl   = 16;
        மாறிலி int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        மாறிலி int64_t cl_per_thread = (total_cl + இலக்கம்_threads - 1) / இலக்கம்_threads;
        மாறிலி int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        எனில் (cl_end > total_cl) {
            cl_end = total_cl;
        }
        எனில் (cl_start >= total_cl) {
            திரும்பு 0;
        }

        மாறிலி int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        எனில் (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        போது (pos < ee) {
            மாறிலி int64_t i1 = row_idx % ne1;
            மாறிலி int64_t i2 = (row_idx / ne1) % ne2;
            மாறிலி int64_t i3 = row_idx / (ne1 * ne2);

            மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            தேர்வு (tri_type) {
                நிலை TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    நிறுத்து;
                நிலை TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    நிறுத்து;
                நிலை TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    நிறுத்து;
                நிலை TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    நிறுத்து;
                இயல்பு:
                    திரும்பு -1;
            }
            எனில் (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            க்கு (int64_t i0 = col; i0 < end_col; i0++) {
                எனில் (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } இல்லையெனில் {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    திரும்பு 0;
}
