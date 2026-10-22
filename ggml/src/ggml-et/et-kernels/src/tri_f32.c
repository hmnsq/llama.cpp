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

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

#घोषणा TRI_TYPE_UPPER_DIAG 0
#घोषणा TRI_TYPE_UPPER      1
#घोषणा TRI_TYPE_LOWER_DIAG 2
#घोषणा TRI_TYPE_LOWER      3

काष्ठा ggml_et_tri_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
    int32_t            tri_type;
};

अटल अंतरभूत पूर्णांक keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    चयन (tri_type) {
        हाल TRI_TYPE_LOWER:
            वापस i0 < i1;
        हाल TRI_TYPE_LOWER_DIAG:
            वापस i0 <= i1;
        हाल TRI_TYPE_UPPER:
            वापस i0 > i1;
        हाल TRI_TYPE_UPPER_DIAG:
            वापस i0 >= i1;
        शेष:
            वापस 0;
    }
}

पूर्णांक entry_point(काष्ठा ggml_et_tri_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0     = &params->src0;
    काष्ठा ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne0 = dst->ne[0];
    स्थिर int64_t ne1 = dst->ne[1];
    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर int64_t ne3 = dst->ne[3];

    स्थिर माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    स्थिर माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    स्थिर int64_t योग_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    अगर (ne0 % 16 == 0) {
        भग्न zero = 0.0f;
        __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        क्रम (int64_t row = thread_id; row < योग_rows; row += अंक_threads) {
            स्थिर int64_t i1 = row % ne1;
            स्थिर int64_t i2 = (row / ne1) % ne2;
            स्थिर int64_t i3 = row / (ne1 * ne2);

            स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i1 * nb01 + i2 * nb02 + i3 * nb03);
            भग्न *       dst_row = (भग्न *) ((अक्षर *) dst_तथ्य + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            चयन (tri_type) {
                हाल TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    अवरोध;
                हाल TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    अवरोध;
                हाल TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    अवरोध;
                हाल TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    अवरोध;
                शेष:
                    वापस -1;
            }
            अगर (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            क्रम (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ अस्थिर("fsw.ps f10, %[d]\n" : [d] "=m"(*(भग्न (*)[8]) & dst_row[i0])::"f10");
            }
            क्रम (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            क्रम (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ अस्थिर(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(भग्न (*)[8]) & dst_row[i0])
                    : [s] "m"(*(स्थिर भग्न (*)[8]) & src_row[i0])
                    : "f11");
            }
            क्रम (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            क्रम (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ अस्थिर("fsw.ps f10, %[d]\n" : [d] "=m"(*(भग्न (*)[8]) & dst_row[i0])::"f10");
            }
            क्रम (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        वापस 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        स्थिर int64_t योग_elements = ne0 * ne1 * ne2 * ne3;
        स्थिर int64_t elems_per_cl   = 16;
        स्थिर int64_t योग_cl       = (योग_elements + elems_per_cl - 1) / elems_per_cl;

        स्थिर int64_t cl_per_thread = (योग_cl + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        अगर (cl_end > योग_cl) {
            cl_end = योग_cl;
        }
        अगर (cl_start >= योग_cl) {
            वापस 0;
        }

        स्थिर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        अगर (ee > योग_elements) {
            ee = योग_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        जबतक (pos < ee) {
            स्थिर int64_t i1 = row_idx % ne1;
            स्थिर int64_t i2 = (row_idx / ne1) % ne2;
            स्थिर int64_t i3 = row_idx / (ne1 * ne2);

            स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            चयन (tri_type) {
                हाल TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    अवरोध;
                हाल TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    अवरोध;
                हाल TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    अवरोध;
                हाल TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    अवरोध;
                शेष:
                    वापस -1;
            }
            अगर (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            क्रम (int64_t i0 = col; i0 < end_col; i0++) {
                अगर (i0 >= keep_start && i0 < keep_end) {
                    dst_तथ्य[pos + (i0 - col)] = src_row[i0];
                } अन्यथा {
                    dst_तथ्य[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    वापस 0;
}
