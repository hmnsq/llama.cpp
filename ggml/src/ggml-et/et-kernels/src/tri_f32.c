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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TRI_TYPE_ਉੱਤਲਾ_DIAG 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TRI_TYPE_ਉੱਤਲਾ      1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TRI_TYPE_ਹੇਠਲਾ_DIAG 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TRI_TYPE_ਹੇਠਲਾ      3

ਬਣਤਰ ggml_et_tri_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
    int32_t            tri_type;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    ਚੋਣ (tri_type) {
        ਹਾਲਤ TRI_TYPE_ਹੇਠਲਾ:
            ਵਾਪਸ i0 < i1;
        ਹਾਲਤ TRI_TYPE_ਹੇਠਲਾ_DIAG:
            ਵਾਪਸ i0 <= i1;
        ਹਾਲਤ TRI_TYPE_ਉੱਤਲਾ:
            ਵਾਪਸ i0 > i1;
        ਹਾਲਤ TRI_TYPE_ਉੱਤਲਾ_DIAG:
            ਵਾਪਸ i0 >= i1;
        ਮੂਲ:
            ਵਾਪਸ 0;
    }
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_tri_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0     = &params->src0;
    ਬਣਤਰ ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne0 = dst->ne[0];
    ਅਚਲ int64_t ne1 = dst->ne[1];
    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ int64_t ne3 = dst->ne[3];

    ਅਚਲ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ਅਚਲ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    ਜੇ (ne0 % 16 == 0) {
        ਦਸ਼ਮਲਵ zero = 0.0f;
        __asm__ ਅਸਥਿਰ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ਲਈ (int64_t ਪੰਕਤੀ = thread_id; ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ += ਅੰਕ_ਧਾਗੇ) {
            ਅਚਲ int64_t i1 = ਪੰਕਤੀ % ne1;
            ਅਚਲ int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
            ਅਚਲ int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_ਸ਼ੁਰੂ, keep_ਸਿਰਾ;
            ਚੋਣ (tri_type) {
                ਹਾਲਤ TRI_TYPE_ਹੇਠਲਾ:
                    keep_ਸ਼ੁਰੂ = 0;
                    keep_ਸਿਰਾ   = i1;
                    ਰੋਕੋ;
                ਹਾਲਤ TRI_TYPE_ਹੇਠਲਾ_DIAG:
                    keep_ਸ਼ੁਰੂ = 0;
                    keep_ਸਿਰਾ   = i1 + 1;
                    ਰੋਕੋ;
                ਹਾਲਤ TRI_TYPE_ਉੱਤਲਾ:
                    keep_ਸ਼ੁਰੂ = i1 + 1;
                    keep_ਸਿਰਾ   = ne0;
                    ਰੋਕੋ;
                ਹਾਲਤ TRI_TYPE_ਉੱਤਲਾ_DIAG:
                    keep_ਸ਼ੁਰੂ = i1;
                    keep_ਸਿਰਾ   = ne0;
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ -1;
            }
            ਜੇ (keep_ਸਿਰਾ > ne0) {
                keep_ਸਿਰਾ = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            ਲਈ (; i0 + 8 <= keep_ਸ਼ੁਰੂ; i0 += 8) {
                __asm__ ਅਸਥਿਰ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[i0])::"f10");
            }
            ਲਈ (; i0 < keep_ਸ਼ੁਰੂ; i0++) {
                dst_ਪੰਕਤੀ[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            ਲਈ (; i0 + 8 <= keep_ਸਿਰਾ; i0 += 8) {
                __asm__ ਅਸਥਿਰ(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[i0])
                    : [s] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ਪੰਕਤੀ[i0])
                    : "f11");
            }
            ਲਈ (; i0 < keep_ਸਿਰਾ; i0++) {
                dst_ਪੰਕਤੀ[i0] = src_ਪੰਕਤੀ[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            ਲਈ (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ ਅਸਥਿਰ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[i0])::"f10");
            }
            ਲਈ (; i0 < ne0; i0++) {
                dst_ਪੰਕਤੀ[i0] = 0.0f;
            }
        }
        ਵਾਪਸ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ਅਚਲ int64_t ਕੁੱਲ_elements = ne0 * ne1 * ne2 * ne3;
        ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl   = 16;
        ਅਚਲ int64_t ਕੁੱਲ_cl       = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

        ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cl + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = thread_id * cl_ਪ੍ਰਤੀ_thread;
        int64_t       cl_ਸਿਰਾ        = cl_ਸ਼ੁਰੂ + cl_ਪ੍ਰਤੀ_thread;
        ਜੇ (cl_ਸਿਰਾ > ਕੁੱਲ_cl) {
            cl_ਸਿਰਾ = ਕੁੱਲ_cl;
        }
        ਜੇ (cl_ਸ਼ੁਰੂ >= ਕੁੱਲ_cl) {
            ਵਾਪਸ 0;
        }

        ਅਚਲ int64_t es = cl_ਸ਼ੁਰੂ * elems_ਪ੍ਰਤੀ_cl;
        int64_t       ee = cl_ਸਿਰਾ * elems_ਪ੍ਰਤੀ_cl;
        ਜੇ (ee > ਕੁੱਲ_elements) {
            ee = ਕੁੱਲ_elements;
        }

        int64_t ਪੰਕਤੀ_idx = es / ne0;
        int64_t ਖਾਨਾ     = es % ne0;

        int64_t pos = es;
        ਜਦੋਂ (pos < ee) {
            ਅਚਲ int64_t i1 = ਪੰਕਤੀ_idx % ne1;
            ਅਚਲ int64_t i2 = (ਪੰਕਤੀ_idx / ne1) % ne2;
            ਅਚਲ int64_t i3 = ਪੰਕਤੀ_idx / (ne1 * ne2);

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t ਪੰਕਤੀ_ਬਚਿਆ   = ne0 - ਖਾਨਾ;
            int64_t chunk_ਬਚਿਆ = ee - pos;
            int64_t n               = ਪੰਕਤੀ_ਬਚਿਆ < chunk_ਬਚਿਆ ? ਪੰਕਤੀ_ਬਚਿਆ : chunk_ਬਚਿਆ;

            int64_t keep_ਸ਼ੁਰੂ, keep_ਸਿਰਾ;
            ਚੋਣ (tri_type) {
                ਹਾਲਤ TRI_TYPE_ਹੇਠਲਾ:
                    keep_ਸ਼ੁਰੂ = 0;
                    keep_ਸਿਰਾ   = i1;
                    ਰੋਕੋ;
                ਹਾਲਤ TRI_TYPE_ਹੇਠਲਾ_DIAG:
                    keep_ਸ਼ੁਰੂ = 0;
                    keep_ਸਿਰਾ   = i1 + 1;
                    ਰੋਕੋ;
                ਹਾਲਤ TRI_TYPE_ਉੱਤਲਾ:
                    keep_ਸ਼ੁਰੂ = i1 + 1;
                    keep_ਸਿਰਾ   = ne0;
                    ਰੋਕੋ;
                ਹਾਲਤ TRI_TYPE_ਉੱਤਲਾ_DIAG:
                    keep_ਸ਼ੁਰੂ = i1;
                    keep_ਸਿਰਾ   = ne0;
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ -1;
            }
            ਜੇ (keep_ਸਿਰਾ > ne0) {
                keep_ਸਿਰਾ = ne0;
            }

            int64_t ਸਿਰਾ_ਖਾਨਾ = ਖਾਨਾ + n;
            ਲਈ (int64_t i0 = ਖਾਨਾ; i0 < ਸਿਰਾ_ਖਾਨਾ; i0++) {
                ਜੇ (i0 >= keep_ਸ਼ੁਰੂ && i0 < keep_ਸਿਰਾ) {
                    dst_ਅੰਕੜੇ[pos + (i0 - ਖਾਨਾ)] = src_ਪੰਕਤੀ[i0];
                } ਨਹੀਂਤਾਂ {
                    dst_ਅੰਕੜੇ[pos + (i0 - ਖਾਨਾ)] = 0.0f;
                }
            }

            pos += n;
            ਖਾਨਾ = 0;
            ਪੰਕਤੀ_idx++;
        }
    }

    ਵਾਪਸ 0;
}
