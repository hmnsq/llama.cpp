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

#परिभाषय TRI_TYPE_ऊर्ध्वस्थम्_DIAG 0
#परिभाषय TRI_TYPE_ऊर्ध्वस्थम्      1
#परिभाषय TRI_TYPE_अधस्तनम्_DIAG 2
#परिभाषय TRI_TYPE_अधस्तनम्      3

संरचना ggml_et_tri_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
    int32_t            tri_type;
};

स्थिर अन्तःस्थ पूर्णाङ्क keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    विकल्प (tri_type) {
        पक्ष TRI_TYPE_अधस्तनम्:
            प्रत्यागम i0 < i1;
        पक्ष TRI_TYPE_अधस्तनम्_DIAG:
            प्रत्यागम i0 <= i1;
        पक्ष TRI_TYPE_ऊर्ध्वस्थम्:
            प्रत्यागम i0 > i1;
        पक्ष TRI_TYPE_ऊर्ध्वस्थम्_DIAG:
            प्रत्यागम i0 >= i1;
        स्वभाव:
            प्रत्यागम 0;
    }
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_tri_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0     = &params->src0;
    संरचना ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne0 = dst->ne[0];
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t ne3 = dst->ne[3];

    ध्रुव size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ध्रुव size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ध्रुव int64_t योग_पङ्क्तिकाः = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    यदि (ne0 % 16 == 0) {
        भिन्न zero = 0.0f;
        __asm__ चञ्चल("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        प्रति (int64_t row = thread_id; row < योग_पङ्क्तिकाः; row += अङ्कः_तन्तवः) {
            ध्रुव int64_t i1 = row % ne1;
            ध्रुव int64_t i2 = (row / ne1) % ne2;
            ध्रुव int64_t i3 = row / (ne1 * ne2);

            ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i1 * nb01 + i2 * nb02 + i3 * nb03);
            भिन्न *       dst_row = (भिन्न *) ((वर्ण *) dst_दत्त + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_अन्तः;
            विकल्प (tri_type) {
                पक्ष TRI_TYPE_अधस्तनम्:
                    keep_start = 0;
                    keep_अन्तः   = i1;
                    भङ्ग;
                पक्ष TRI_TYPE_अधस्तनम्_DIAG:
                    keep_start = 0;
                    keep_अन्तः   = i1 + 1;
                    भङ्ग;
                पक्ष TRI_TYPE_ऊर्ध्वस्थम्:
                    keep_start = i1 + 1;
                    keep_अन्तः   = ne0;
                    भङ्ग;
                पक्ष TRI_TYPE_ऊर्ध्वस्थम्_DIAG:
                    keep_start = i1;
                    keep_अन्तः   = ne0;
                    भङ्ग;
                स्वभाव:
                    प्रत्यागम -1;
            }
            यदि (keep_अन्तः > ne0) {
                keep_अन्तः = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            प्रति (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ चञ्चल("fsw.ps f10, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i0])::"f10");
            }
            प्रति (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            प्रति (; i0 + 8 <= keep_अन्तः; i0 += 8) {
                __asm__ चञ्चल(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i0])
                    : [s] "m"(*(ध्रुव भिन्न (*)[8]) & src_row[i0])
                    : "f11");
            }
            प्रति (; i0 < keep_अन्तः; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            प्रति (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ चञ्चल("fsw.ps f10, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i0])::"f10");
            }
            प्रति (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        प्रत्यागम 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ध्रुव int64_t योग_elements = ne0 * ne1 * ne2 * ne3;
        ध्रुव int64_t elems_प्रत्येकम्_cl   = 16;
        ध्रुव int64_t योग_cl       = (योग_elements + elems_प्रत्येकम्_cl - 1) / elems_प्रत्येकम्_cl;

        ध्रुव int64_t cl_प्रत्येकम्_thread = (योग_cl + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
        ध्रुव int64_t cl_start      = thread_id * cl_प्रत्येकम्_thread;
        int64_t       cl_अन्तः        = cl_start + cl_प्रत्येकम्_thread;
        यदि (cl_अन्तः > योग_cl) {
            cl_अन्तः = योग_cl;
        }
        यदि (cl_start >= योग_cl) {
            प्रत्यागम 0;
        }

        ध्रुव int64_t es = cl_start * elems_प्रत्येकम्_cl;
        int64_t       ee = cl_अन्तः * elems_प्रत्येकम्_cl;
        यदि (ee > योग_elements) {
            ee = योग_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        यावत् (pos < ee) {
            ध्रुव int64_t i1 = row_idx % ne1;
            ध्रुव int64_t i2 = (row_idx / ne1) % ne2;
            ध्रुव int64_t i3 = row_idx / (ne1 * ne2);

            ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_अन्तः;
            विकल्प (tri_type) {
                पक्ष TRI_TYPE_अधस्तनम्:
                    keep_start = 0;
                    keep_अन्तः   = i1;
                    भङ्ग;
                पक्ष TRI_TYPE_अधस्तनम्_DIAG:
                    keep_start = 0;
                    keep_अन्तः   = i1 + 1;
                    भङ्ग;
                पक्ष TRI_TYPE_ऊर्ध्वस्थम्:
                    keep_start = i1 + 1;
                    keep_अन्तः   = ne0;
                    भङ्ग;
                पक्ष TRI_TYPE_ऊर्ध्वस्थम्_DIAG:
                    keep_start = i1;
                    keep_अन्तः   = ne0;
                    भङ्ग;
                स्वभाव:
                    प्रत्यागम -1;
            }
            यदि (keep_अन्तः > ne0) {
                keep_अन्तः = ne0;
            }

            int64_t अन्तः_col = col + n;
            प्रति (int64_t i0 = col; i0 < अन्तः_col; i0++) {
                यदि (i0 >= keep_start && i0 < keep_अन्तः) {
                    dst_दत्त[pos + (i0 - col)] = src_row[i0];
                } अन्यथा {
                    dst_दत्त[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    प्रत्यागम 0;
}
