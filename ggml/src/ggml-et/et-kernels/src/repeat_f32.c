//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>
#ضمن <string.h>

بنية ggml_et_repeat_params {
    بنية ggml_tensor src0;  // F32 input tensor (tile)
    بنية ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ساكن مضمن فراغ copy_صف_aligned(عائم * dst, ثابت عائم * src, int32_t n) {
    لكل (int32_t i = 0; i < n; i += 8) {
        __asm__ متطاير(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(عائم (*)[8]) & dst[i])
            : [src_vec] "m"(*(ثابت عائم (*)[8]) & src[i])
            : "f11");
    }
}

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
ساكن مضمن فراغ نشر_scalar_aligned(عائم * dst, عائم val, int32_t n) {
    __asm__ متطاير("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    لكل (int32_t i = 0; i < n; i += 8) {
        __asm__ متطاير("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(عائم (*)[8]) & dst[i])::"f11");
    }
}

صحيح داخلا_نقطو(بنية ggml_et_repeat_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
        أرجع -1;
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
    ثابت int64_t ڪل_صفون = ne1 * ne2 * ne3;

    لكل (int64_t صف = thread_id; صف < ڪل_صفون; صف += انگ_ڌاڳا) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = صف % ne1;
        int64_t i2 = (صف / ne1) % ne2;
        int64_t i3 = صف / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + k1 * nb01 + k2 * nb02 + k3 * nb03);
        عائم *       dst_صف = (عائم *) ((حرف *) dst_معلومات + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        إذا (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            نشر_scalar_aligned(dst_صف, *src_صف, (int32_t) ne0);
        } وإلا إذا (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_صف_aligned(dst_صف, src_صف, (int32_t) ne00);
        } وإلا {
            // Tile ne00-sized chunks across dim 0
            لكل (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_صف_aligned(dst_صف + i0 * ne00, src_صف, (int32_t) ne00);
            }
        }
    }

    أرجع 0;
}
