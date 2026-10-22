//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>
#समावेश <string.h>

काष्ठा ggml_et_repeat_params {
    काष्ठा ggml_tensor src0;  // F32 input tensor (tile)
    काष्ठा ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
अटल अंतरभूत व्योम copy_row_aligned(भग्न * dst, स्थिर भग्न * src, int32_t n) {
    क्रम (int32_t i = 0; i < n; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(भग्न (*)[8]) & dst[i])
            : [src_vec] "m"(*(स्थिर भग्न (*)[8]) & src[i])
            : "f11");
    }
}

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
अटल अंतरभूत व्योम broadcast_scalar_aligned(भग्न * dst, भग्न val, int32_t n) {
    __asm__ अस्थिर("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    क्रम (int32_t i = 0; i < n; i += 8) {
        __asm__ अस्थिर("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(भग्न (*)[8]) & dst[i])::"f11");
    }
}

पूर्णांक entry_point(काष्ठा ggml_et_repeat_params * params, व्योम * env) {
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

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    स्थिर int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    स्थिर माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    स्थिर माप_प्रकार dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    स्थिर int32_t nr0 = (int32_t) (ne0 / ne00);
    स्थिर int32_t nr1 = (int32_t) (ne1 / ne01);
    स्थिर int32_t nr2 = (int32_t) (ne2 / ne02);
    स्थिर int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    स्थिर int64_t योग_rows = ne1 * ne2 * ne3;

    क्रम (int64_t row = thread_id; row < योग_rows; row += अंक_threads) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + k1 * nb01 + k2 * nb02 + k3 * nb03);
        भग्न *       dst_row = (भग्न *) ((अक्षर *) dst_तथ्य + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        अगर (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            broadcast_scalar_aligned(dst_row, *src_row, (int32_t) ne0);
        } अन्यथा अगर (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_row_aligned(dst_row, src_row, (int32_t) ne00);
        } अन्यथा {
            // Tile ne00-sized chunks across dim 0
            क्रम (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_row_aligned(dst_row + i0 * ne00, src_row, (int32_t) ne00);
            }
        }
    }

    वापस 0;
}
