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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

સંરચના ggml_et_tri_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
    int32_t            tri_type;
};

સ્થિર અંતઃરેખ પૂર્ણાંક keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    પસંદગી (tri_type) {
        સ્થિતિ TRI_TYPE_LOWER:
            પાછા i0 < i1;
        સ્થિતિ TRI_TYPE_LOWER_DIAG:
            પાછા i0 <= i1;
        સ્થિતિ TRI_TYPE_UPPER:
            પાછા i0 > i1;
        સ્થિતિ TRI_TYPE_UPPER_DIAG:
            પાછા i0 >= i1;
        મૂળભૂત:
            પાછા 0;
    }
}

પૂર્ણાંક entry_point(સંરચના ggml_et_tri_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0     = &params->src0;
    સંરચના ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne0 = dst->ne[0];
    અચલ int64_t ne1 = dst->ne[1];
    અચલ int64_t ne2 = dst->ne[2];
    અચલ int64_t ne3 = dst->ne[3];

    અચલ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    અચલ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    અચલ int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    જો (ne0 % 16 == 0) {
        દશાંશ zero = 0.0f;
        __asm__ અસ્થિર("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        માટે (int64_t row = thread_id; row < total_rows; row += અંક_threads) {
            અચલ int64_t i1 = row % ne1;
            અચલ int64_t i2 = (row / ne1) % ne2;
            અચલ int64_t i3 = row / (ne1 * ne2);

            અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            દશાંશ *       dst_row = (દશાંશ *) ((અક્ષર *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            પસંદગી (tri_type) {
                સ્થિતિ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    તોડો;
                સ્થિતિ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    તોડો;
                સ્થિતિ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    તોડો;
                સ્થિતિ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    તોડો;
                મૂળભૂત:
                    પાછા -1;
            }
            જો (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            માટે (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ અસ્થિર("fsw.ps f10, %[d]\n" : [d] "=m"(*(દશાંશ (*)[8]) & dst_row[i0])::"f10");
            }
            માટે (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            માટે (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ અસ્થિર(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(દશાંશ (*)[8]) & dst_row[i0])
                    : [s] "m"(*(અચલ દશાંશ (*)[8]) & src_row[i0])
                    : "f11");
            }
            માટે (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            માટે (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ અસ્થિર("fsw.ps f10, %[d]\n" : [d] "=m"(*(દશાંશ (*)[8]) & dst_row[i0])::"f10");
            }
            માટે (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        પાછા 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        અચલ int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        અચલ int64_t elems_per_cl   = 16;
        અચલ int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        અચલ int64_t cl_per_thread = (total_cl + અંક_threads - 1) / અંક_threads;
        અચલ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        જો (cl_end > total_cl) {
            cl_end = total_cl;
        }
        જો (cl_start >= total_cl) {
            પાછા 0;
        }

        અચલ int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        જો (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        જ્યારે (pos < ee) {
            અચલ int64_t i1 = row_idx % ne1;
            અચલ int64_t i2 = (row_idx / ne1) % ne2;
            અચલ int64_t i3 = row_idx / (ne1 * ne2);

            અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            પસંદગી (tri_type) {
                સ્થિતિ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    તોડો;
                સ્થિતિ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    તોડો;
                સ્થિતિ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    તોડો;
                સ્થિતિ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    તોડો;
                મૂળભૂત:
                    પાછા -1;
            }
            જો (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            માટે (int64_t i0 = col; i0 < end_col; i0++) {
                જો (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } નહીંતર {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    પાછા 0;
}
