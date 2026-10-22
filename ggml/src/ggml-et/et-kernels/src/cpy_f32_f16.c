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

ಸಂರಚನಾ ggml_et_cont_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_cont_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env || !params) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ವರ್ಣ * src_data = (ಧ್ರುವ ವರ್ಣ *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    ಯದಿ (!src_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t ne00 = src0->ne[0];
    ಧ್ರುವ int64_t ne01 = src0->ne[1];
    ಧ್ರುವ int64_t ne02 = src0->ne[2];
    ಧ್ರುವ int64_t ne03 = src0->ne[3];

    ಧ್ರುವ int64_t nb00 = src0->nb[0];
    ಧ್ರುವ int64_t nb01 = src0->nb[1];
    ಧ್ರುವ int64_t nb02 = src0->nb[2];
    ಧ್ರುವ int64_t nb03 = src0->nb[3];

    ಧ್ರುವ int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    ಯದಿ (total_elements == 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Check if src is contiguous F32
    ಧ್ರುವ bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ಧ್ರುವ int64_t elems_per_cl = 32;
    ಧ್ರುವ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ಧ್ರುವ int64_t cl_per_thread = (total_cl + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
    ಧ್ರುವ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ಯದಿ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ಯದಿ (cl_start >= total_cl) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ಯದಿ (ee > total_elements) {
        ee = total_elements;
    }

    ಯದಿ (src_contiguous) {
        // Fast path: src is contiguous F32
        ಧ್ರುವ ಭಿನ್ನ * src_f32 = (ಧ್ರುವ ಭಿನ್ನ *) src_data;
        ಪ್ರತಿ (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } ಅನ್ಯಥಾ {
        // General path: stride-aware read
        ಪ್ರತಿ (int64_t idx = es; idx < ee; ++idx) {
            ಧ್ರುವ int64_t i00  = idx % ne00;
            ಧ್ರುವ int64_t rem1 = idx / ne00;
            ಧ್ರುವ int64_t i01  = rem1 % ne01;
            ಧ್ರುವ int64_t rem2 = rem1 / ne01;
            ಧ್ರುವ int64_t i02  = rem2 % ne02;
            ಧ್ರುವ int64_t i03  = rem2 / ne02;

            ಧ್ರುವ ಭಿನ್ನ val = *(ಧ್ರುವ ಭಿನ್ನ *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
