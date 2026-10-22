//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_cont_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_cont_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env || !params) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    uint16_t *   dst_ದತ್ತಾಂಶ = (uint16_t *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ int64_t nb00 = src0->nb[0];
    ಸ್ಥಿರಾಂಕ int64_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb03 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = ne00 * ne01 * ne02 * ne03;

    ಆದರೆ (ಒಟ್ಟು_elements == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Check if src is contiguous F32
    ಸ್ಥಿರಾಂಕ bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl = 32;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl     = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

    ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cl + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
    int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
    ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cl) {
        cl_ಕೊನೆ = ಒಟ್ಟು_cl;
    }
    ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cl) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ int64_t es = cl_ಆರಂಭ * elems_ಪ್ರತಿ_cl;
    int64_t       ee = cl_ಕೊನೆ * elems_ಪ್ರತಿ_cl;
    ಆದರೆ (ee > ಒಟ್ಟು_elements) {
        ee = ಒಟ್ಟು_elements;
    }

    ಆದರೆ (src_contiguous) {
        // Fast path: src is contiguous F32
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_f32 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_ದತ್ತಾಂಶ;
        ಗಾಗಿ (int64_t i = es; i < ee; ++i) {
            dst_ದತ್ತಾಂಶ[i] = fp32_ಗಮ್ಯ_fp16(src_f32[i]);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // General path: stride-aware read
        ಗಾಗಿ (int64_t idx = es; idx < ee; ++idx) {
            ಸ್ಥಿರಾಂಕ int64_t i00  = idx % ne00;
            ಸ್ಥಿರಾಂಕ int64_t rem1 = idx / ne00;
            ಸ್ಥಿರಾಂಕ int64_t i01  = rem1 % ne01;
            ಸ್ಥಿರಾಂಕ int64_t rem2 = rem1 / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i02  = rem2 % ne02;
            ಸ್ಥಿರಾಂಕ int64_t i03  = rem2 / ne02;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (src_ದತ್ತಾಂಶ + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_ದತ್ತಾಂಶ[idx]   = fp32_ಗಮ್ಯ_fp16(val);
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
