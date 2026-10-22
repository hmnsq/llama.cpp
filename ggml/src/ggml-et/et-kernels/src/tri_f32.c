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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

संरचना ggml_et_tri_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
    int32_t            tri_type;
};

स्थिर अन्तःस्थ पूर्णाङ्क keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    विकल्प (tri_type) {
        पक्ष TRI_TYPE_LOWER:
            फर्काऊ i0 < i1;
        पक्ष TRI_TYPE_LOWER_DIAG:
            फर्काऊ i0 <= i1;
        पक्ष TRI_TYPE_UPPER:
            फर्काऊ i0 > i1;
        पक्ष TRI_TYPE_UPPER_DIAG:
            फर्काऊ i0 >= i1;
        स्वभाव:
            फर्काऊ 0;
    }
}

पूर्णाङ्क entry_point(संरचना ggml_et_tri_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * src0     = &params->src0;
    संरचना ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        फर्काऊ -1;
    }

    अचर int64_t ne0 = dst->ne[0];
    अचर int64_t ne1 = dst->ne[1];
    अचर int64_t ne2 = dst->ne[2];
    अचर int64_t ne3 = dst->ne[3];

    अचर माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    अचर माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    अचर int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    यदि (ne0 % 16 == 0) {
        भिन्न zero = 0.0f;
        __asm__ चञ्चल("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        लागि (int64_t row = thread_id; row < total_rows; row += अङ्क_threads) {
            अचर int64_t i1 = row % ne1;
            अचर int64_t i2 = (row / ne1) % ne2;
            अचर int64_t i3 = row / (ne1 * ne2);

            अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            भिन्न *       dst_row = (भिन्न *) ((वर्ण *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            विकल्प (tri_type) {
                पक्ष TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    रोक;
                पक्ष TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    रोक;
                पक्ष TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    रोक;
                पक्ष TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    रोक;
                स्वभाव:
                    फर्काऊ -1;
            }
            यदि (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            लागि (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ चञ्चल("fsw.ps f10, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i0])::"f10");
            }
            लागि (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            लागि (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ चञ्चल(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i0])
                    : [s] "m"(*(अचर भिन्न (*)[8]) & src_row[i0])
                    : "f11");
            }
            लागि (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            लागि (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ चञ्चल("fsw.ps f10, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i0])::"f10");
            }
            लागि (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        फर्काऊ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        अचर int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        अचर int64_t elems_per_cl   = 16;
        अचर int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        अचर int64_t cl_per_thread = (total_cl + अङ्क_threads - 1) / अङ्क_threads;
        अचर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        यदि (cl_end > total_cl) {
            cl_end = total_cl;
        }
        यदि (cl_start >= total_cl) {
            फर्काऊ 0;
        }

        अचर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        यदि (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        जबसम्म (pos < ee) {
            अचर int64_t i1 = row_idx % ne1;
            अचर int64_t i2 = (row_idx / ne1) % ne2;
            अचर int64_t i3 = row_idx / (ne1 * ne2);

            अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            विकल्प (tri_type) {
                पक्ष TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    रोक;
                पक्ष TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    रोक;
                पक्ष TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    रोक;
                पक्ष TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    रोक;
                स्वभाव:
                    फर्काऊ -1;
            }
            यदि (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            लागि (int64_t i0 = col; i0 < end_col; i0++) {
                यदि (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } अन्यथा {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    फर्काऊ 0;
}
