//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <string.h>

কাঠামো ggml_et_repeat_params {
    কাঠামো ggml_tensor src0;  // F32 input tensor (tile)
    কাঠামো ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
স্থির অন্তঃরেখ শূন্য copy_row_aligned(দশমিক * dst, ধ্রুবক দশমিক * src, int32_t n) {
    জন্য (int32_t i = 0; i < n; i += 8) {
        __asm__ অস্থির(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(দশমিক (*)[8]) & dst[i])
            : [src_vec] "m"(*(ধ্রুবক দশমিক (*)[8]) & src[i])
            : "f11");
    }
}

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
স্থির অন্তঃরেখ শূন্য broadcast_scalar_aligned(দশমিক * dst, দশমিক val, int32_t n) {
    __asm__ অস্থির("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    জন্য (int32_t i = 0; i < n; i += 8) {
        __asm__ অস্থির("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(দশমিক (*)[8]) & dst[i])::"f11");
    }
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_repeat_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ধ্রুবক int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    ধ্রুবক size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    ধ্রুবক size_t dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    ধ্রুবক int32_t nr0 = (int32_t) (ne0 / ne00);
    ধ্রুবক int32_t nr1 = (int32_t) (ne1 / ne01);
    ধ্রুবক int32_t nr2 = (int32_t) (ne2 / ne02);
    ধ্রুবক int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    ধ্রুবক int64_t যোগ_rows = ne1 * ne2 * ne3;

    জন্য (int64_t row = thread_id; row < যোগ_rows; row += অঙ্ক_threads) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + k1 * nb01 + k2 * nb02 + k3 * nb03);
        দশমিক *       dst_row = (দশমিক *) ((অক্ষর *) dst_তথ্য + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        যদি (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            broadcast_scalar_aligned(dst_row, *src_row, (int32_t) ne0);
        } অন্যথা যদি (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_row_aligned(dst_row, src_row, (int32_t) ne00);
        } অন্যথা {
            // Tile ne00-sized chunks across dim 0
            জন্য (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_row_aligned(dst_row + i0 * ne00, src_row, (int32_t) ne00);
            }
        }
    }

    ফেরত 0;
}
