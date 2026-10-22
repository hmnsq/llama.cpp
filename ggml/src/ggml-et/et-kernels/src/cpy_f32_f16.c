//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_cont_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_cont_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env || !params) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਅੱਖਰ * src_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    uint16_t *   dst_ਅੰਕੜੇ = (uint16_t *) dst->ਅੰਕੜੇ;

    ਜੇ (!src_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    ਅਚਲ int64_t nb00 = src0->nb[0];
    ਅਚਲ int64_t nb01 = src0->nb[1];
    ਅਚਲ int64_t nb02 = src0->nb[2];
    ਅਚਲ int64_t nb03 = src0->nb[3];

    ਅਚਲ int64_t ਕੁੱਲ_elements = ne00 * ne01 * ne02 * ne03;

    ਜੇ (ਕੁੱਲ_elements == 0) {
        ਵਾਪਸ 0;
    }

    // Check if src is contiguous F32
    ਅਚਲ bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl = 32;
    ਅਚਲ int64_t ਕੁੱਲ_cl     = (ਕੁੱਲ_elements + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

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

    ਜੇ (src_contiguous) {
        // Fast path: src is contiguous F32
        ਅਚਲ ਦਸ਼ਮਲਵ * src_f32 = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_ਅੰਕੜੇ;
        ਲਈ (int64_t i = es; i < ee; ++i) {
            dst_ਅੰਕੜੇ[i] = fp32_ਮੰਜ਼ਿਲ_fp16(src_f32[i]);
        }
    } ਨਹੀਂਤਾਂ {
        // General path: stride-aware read
        ਲਈ (int64_t idx = es; idx < ee; ++idx) {
            ਅਚਲ int64_t i00  = idx % ne00;
            ਅਚਲ int64_t rem1 = idx / ne00;
            ਅਚਲ int64_t i01  = rem1 % ne01;
            ਅਚਲ int64_t rem2 = rem1 / ne01;
            ਅਚਲ int64_t i02  = rem2 % ne02;
            ਅਚਲ int64_t i03  = rem2 / ne02;

            ਅਚਲ ਦਸ਼ਮਲਵ val = *(ਅਚਲ ਦਸ਼ਮਲਵ *) (src_ਅੰਕੜੇ + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_ਅੰਕੜੇ[idx]   = fp32_ਮੰਜ਼ਿਲ_fp16(val);
        }
    }

    ਵਾਪਸ 0;
}
