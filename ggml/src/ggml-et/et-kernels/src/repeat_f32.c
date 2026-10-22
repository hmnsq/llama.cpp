//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

ସଂରଚନା ggml_et_repeat_params {
    ସଂରଚନା ggml_tensor src0;  // F32 input tensor (tile)
    ସଂରଚନା ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ copy_row_aligned(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, int32_t n) {
    ପାଇଁ (int32_t i = 0; i < n; i += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ଦଶମିକ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src[i])
            : "f11");
    }
}

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ broadcast_scalar_aligned(ଦଶମିକ * dst, ଦଶମିକ val, int32_t n) {
    __asm__ ଅସ୍ଥିର("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    ପାଇଁ (int32_t i = 0; i < n; i += 8) {
        __asm__ ଅସ୍ଥିର("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(ଦଶମିକ (*)[8]) & dst[i])::"f11");
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_repeat_params * params, ଶୂନ୍ୟ * env) {
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

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ଅଚଳ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    ଅଚଳ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    ଅଚଳ size_t dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    ଅଚଳ int32_t nr0 = (int32_t) (ne0 / ne00);
    ଅଚଳ int32_t nr1 = (int32_t) (ne1 / ne01);
    ଅଚଳ int32_t nr2 = (int32_t) (ne2 / ne02);
    ଅଚଳ int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    ଅଚଳ int64_t total_rows = ne1 * ne2 * ne3;

    ପାଇଁ (int64_t row = thread_id; row < total_rows; row += ଅଙ୍କ_threads) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + k1 * nb01 + k2 * nb02 + k3 * nb03);
        ଦଶମିକ *       dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ଯଦି (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            broadcast_scalar_aligned(dst_row, *src_row, (int32_t) ne0);
        } ଅନ୍ୟଥା ଯଦି (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_row_aligned(dst_row, src_row, (int32_t) ne00);
        } ଅନ୍ୟଥା {
            // Tile ne00-sized chunks across dim 0
            ପାଇଁ (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_row_aligned(dst_row + i0 * ne00, src_row, (int32_t) ne00);
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
