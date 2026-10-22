//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

संरचना ggml_et_repeat_params {
    संरचना ggml_tensor src0;  // F32 input tensor (tile)
    संरचना ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
स्थिर अन्तःस्थ शून्य copy_row_aligned(भिन्न * dst, ध्रुव भिन्न * src, int32_t n) {
    प्रति (int32_t i = 0; i < n; i += 8) {
        __asm__ चञ्चल(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(भिन्न (*)[8]) & dst[i])
            : [src_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src[i])
            : "f11");
    }
}

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
स्थिर अन्तःस्थ शून्य broadcast_scalar_aligned(भिन्न * dst, भिन्न val, int32_t n) {
    __asm__ चञ्चल("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    प्रति (int32_t i = 0; i < n; i += 8) {
        __asm__ चञ्चल("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(भिन्न (*)[8]) & dst[i])::"f11");
    }
}

पूर्णाङ्क entry_point(संरचना ggml_et_repeat_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ध्रुव int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    ध्रुव माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    ध्रुव माप_प्रकार dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    ध्रुव int32_t nr0 = (int32_t) (ne0 / ne00);
    ध्रुव int32_t nr1 = (int32_t) (ne1 / ne01);
    ध्रुव int32_t nr2 = (int32_t) (ne2 / ne02);
    ध्रुव int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    ध्रुव int64_t total_rows = ne1 * ne2 * ne3;

    प्रति (int64_t row = thread_id; row < total_rows; row += अंक_threads) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + k1 * nb01 + k2 * nb02 + k3 * nb03);
        भिन्न *       dst_row = (भिन्न *) ((वर्ण *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        यदि (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            broadcast_scalar_aligned(dst_row, *src_row, (int32_t) ne0);
        } अन्यथा यदि (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_row_aligned(dst_row, src_row, (int32_t) ne00);
        } अन्यथा {
            // Tile ne00-sized chunks across dim 0
            प्रति (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_row_aligned(dst_row + i0 * ne00, src_row, (int32_t) ne00);
            }
        }
    }

    प्रत्यागम 0;
}
