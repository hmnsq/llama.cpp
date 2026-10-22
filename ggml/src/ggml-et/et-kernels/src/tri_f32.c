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

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

ഘടന ggml_et_tri_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor dst;
    int32_t            tri_type;
};

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    തിരഞ്ഞെടുപ്പ് (tri_type) {
        അവസ്ഥ TRI_TYPE_LOWER:
            തിരികെ i0 < i1;
        അവസ്ഥ TRI_TYPE_LOWER_DIAG:
            തിരികെ i0 <= i1;
        അവസ്ഥ TRI_TYPE_UPPER:
            തിരികെ i0 > i1;
        അവസ്ഥ TRI_TYPE_UPPER_DIAG:
            തിരികെ i0 >= i1;
        സ്ഥിരസ്ഥിതി:
            തിരികെ 0;
    }
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_tri_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0     = &params->src0;
    ഘടന ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t ne0 = dst->ne[0];
    സ്ഥിരം int64_t ne1 = dst->ne[1];
    സ്ഥിരം int64_t ne2 = dst->ne[2];
    സ്ഥിരം int64_t ne3 = dst->ne[3];

    സ്ഥിരം size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    സ്ഥിരം size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    സ്ഥിരം int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    എങ്കിൽ (ne0 % 16 == 0) {
        ദശാംശം zero = 0.0f;
        __asm__ അസ്ഥിര("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        വേണ്ടി (int64_t row = thread_id; row < total_rows; row += അക്കം_threads) {
            സ്ഥിരം int64_t i1 = row % ne1;
            സ്ഥിരം int64_t i2 = (row / ne1) % ne2;
            സ്ഥിരം int64_t i3 = row / (ne1 * ne2);

            സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ദശാംശം *       dst_row = (ദശാംശം *) ((അക്ഷരം *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            തിരഞ്ഞെടുപ്പ് (tri_type) {
                അവസ്ഥ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    നിർത്തുക;
                അവസ്ഥ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    നിർത്തുക;
                അവസ്ഥ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    നിർത്തുക;
                അവസ്ഥ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    തിരികെ -1;
            }
            എങ്കിൽ (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            വേണ്ടി (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ അസ്ഥിര("fsw.ps f10, %[d]\n" : [d] "=m"(*(ദശാംശം (*)[8]) & dst_row[i0])::"f10");
            }
            വേണ്ടി (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            വേണ്ടി (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ അസ്ഥിര(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(ദശാംശം (*)[8]) & dst_row[i0])
                    : [s] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src_row[i0])
                    : "f11");
            }
            വേണ്ടി (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            വേണ്ടി (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ അസ്ഥിര("fsw.ps f10, %[d]\n" : [d] "=m"(*(ദശാംശം (*)[8]) & dst_row[i0])::"f10");
            }
            വേണ്ടി (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        തിരികെ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        സ്ഥിരം int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        സ്ഥിരം int64_t elems_per_cl   = 16;
        സ്ഥിരം int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        സ്ഥിരം int64_t cl_per_thread = (total_cl + അക്കം_threads - 1) / അക്കം_threads;
        സ്ഥിരം int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        എങ്കിൽ (cl_end > total_cl) {
            cl_end = total_cl;
        }
        എങ്കിൽ (cl_start >= total_cl) {
            തിരികെ 0;
        }

        സ്ഥിരം int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        എങ്കിൽ (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        ആകുമ്പോൾ (pos < ee) {
            സ്ഥിരം int64_t i1 = row_idx % ne1;
            സ്ഥിരം int64_t i2 = (row_idx / ne1) % ne2;
            സ്ഥിരം int64_t i3 = row_idx / (ne1 * ne2);

            സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            തിരഞ്ഞെടുപ്പ് (tri_type) {
                അവസ്ഥ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    നിർത്തുക;
                അവസ്ഥ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    നിർത്തുക;
                അവസ്ഥ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    നിർത്തുക;
                അവസ്ഥ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    തിരികെ -1;
            }
            എങ്കിൽ (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            വേണ്ടി (int64_t i0 = col; i0 < end_col; i0++) {
                എങ്കിൽ (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } അല്ലെങ്കിൽ {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    തിരികെ 0;
}
