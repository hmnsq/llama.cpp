//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>
#شامل <string.h>

ساخت ggml_et_repeat_params {
    ساخت ggml_tensor src0;  // F32 input tensor (tile)
    ساخت ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ساکن اندرونی خالی copy_صف_aligned(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [src_vec] "m"(*(ثابت اعشاری (*)[8]) & src[i])
            : "f11");
    }
}

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
ساکن اندرونی خالی نشر_scalar_aligned(اعشاری * dst, اعشاری val, int32_t n) {
    __asm__ متلون("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(اعشاری (*)[8]) & dst[i])::"f11");
    }
}

صحیح اندراج_نقطہ(ساخت ggml_et_repeat_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ثابت int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    ثابت size_t dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    ثابت int32_t nr0 = (int32_t) (ne0 / ne00);
    ثابت int32_t nr1 = (int32_t) (ne1 / ne01);
    ثابت int32_t nr2 = (int32_t) (ne2 / ne02);
    ثابت int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    ثابت int64_t کل_صفیں = ne1 * ne2 * ne3;

    برائے (int64_t صف = thread_id; صف < کل_صفیں; صف += عدد_دھاگے) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = صف % ne1;
        int64_t i2 = (صف / ne1) % ne2;
        int64_t i3 = صف / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + k1 * nb01 + k2 * nb02 + k3 * nb03);
        اعشاری *       dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        اگر (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            نشر_scalar_aligned(dst_صف, *src_صف, (int32_t) ne0);
        } ورنہ اگر (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_صف_aligned(dst_صف, src_صف, (int32_t) ne00);
        } ورنہ {
            // Tile ne00-sized chunks across dim 0
            برائے (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_صف_aligned(dst_صف + i0 * ne00, src_صف, (int32_t) ne00);
            }
        }
    }

    واپس 0;
}
