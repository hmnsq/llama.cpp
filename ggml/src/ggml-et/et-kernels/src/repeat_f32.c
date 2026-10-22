//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <string.h>

ഘടന ggml_et_repeat_params {
    ഘടന ggml_tensor src0;  // F32 input tensor (tile)
    ഘടന ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
സ്ഥിതം വരിയിൽ ശൂന്യം copy_row_aligned(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, int32_t n) {
    വേണ്ടി (int32_t i = 0; i < n; i += 8) {
        __asm__ അസ്ഥിര(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ദശാംശം (*)[8]) & dst[i])
            : [src_vec] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src[i])
            : "f11");
    }
}

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
സ്ഥിതം വരിയിൽ ശൂന്യം broadcast_scalar_aligned(ദശാംശം * dst, ദശാംശം val, int32_t n) {
    __asm__ അസ്ഥിര("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    വേണ്ടി (int32_t i = 0; i < n; i += 8) {
        __asm__ അസ്ഥിര("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(ദശാംശം (*)[8]) & dst[i])::"f11");
    }
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_repeat_params * params, ശൂന്യം * env) {
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

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    സ്ഥിരം int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    സ്ഥിരം size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    സ്ഥിരം size_t dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    സ്ഥിരം int32_t nr0 = (int32_t) (ne0 / ne00);
    സ്ഥിരം int32_t nr1 = (int32_t) (ne1 / ne01);
    സ്ഥിരം int32_t nr2 = (int32_t) (ne2 / ne02);
    സ്ഥിരം int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    സ്ഥിരം int64_t total_rows = ne1 * ne2 * ne3;

    വേണ്ടി (int64_t row = thread_id; row < total_rows; row += അക്കം_threads) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + k1 * nb01 + k2 * nb02 + k3 * nb03);
        ദശാംശം *       dst_row = (ദശാംശം *) ((അക്ഷരം *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        എങ്കിൽ (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            broadcast_scalar_aligned(dst_row, *src_row, (int32_t) ne0);
        } അല്ലെങ്കിൽ എങ്കിൽ (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_row_aligned(dst_row, src_row, (int32_t) ne00);
        } അല്ലെങ്കിൽ {
            // Tile ne00-sized chunks across dim 0
            വേണ്ടി (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_row_aligned(dst_row + i0 * ne00, src_row, (int32_t) ne00);
            }
        }
    }

    തിരികെ 0;
}
