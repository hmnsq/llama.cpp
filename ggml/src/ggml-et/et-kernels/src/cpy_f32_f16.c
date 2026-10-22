//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

संरचना ggml_et_cont_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
};

पूर्णाङ्क entry_point(संरचना ggml_et_cont_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जइ (!kernel_env || !params) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्कः_threads = get_अङ्कः_threads(kernel_env->shire_mask);

    जइ (thread_id < 0) {
        प्रत्यागम 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जइ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        प्रत्यागम -1;
    }

    ध्रुव वर्ण * src_data = (ध्रुव वर्ण *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    जइ (!src_data || !dst_data) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];
    ध्रुव int64_t ne02 = src0->ne[2];
    ध्रुव int64_t ne03 = src0->ne[3];

    ध्रुव int64_t nb00 = src0->nb[0];
    ध्रुव int64_t nb01 = src0->nb[1];
    ध्रुव int64_t nb02 = src0->nb[2];
    ध्रुव int64_t nb03 = src0->nb[3];

    ध्रुव int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    जइ (total_elements == 0) {
        प्रत्यागम 0;
    }

    // Check if src is contiguous F32
    ध्रुव bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ध्रुव int64_t elems_per_cl = 32;
    ध्रुव int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ध्रुव int64_t cl_per_thread = (total_cl + अङ्कः_threads - 1) / अङ्कः_threads;
    ध्रुव int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    जइ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    जइ (cl_start >= total_cl) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    जइ (ee > total_elements) {
        ee = total_elements;
    }

    जइ (src_contiguous) {
        // Fast path: src is contiguous F32
        ध्रुव भिन्न * src_f32 = (ध्रुव भिन्न *) src_data;
        प्रति (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } अण्णहा {
        // General path: stride-aware read
        प्रति (int64_t idx = es; idx < ee; ++idx) {
            ध्रुव int64_t i00  = idx % ne00;
            ध्रुव int64_t rem1 = idx / ne00;
            ध्रुव int64_t i01  = rem1 % ne01;
            ध्रुव int64_t rem2 = rem1 / ne01;
            ध्रुव int64_t i02  = rem2 % ne02;
            ध्रुव int64_t i03  = rem2 / ne02;

            ध्रुव भिन्न val = *(ध्रुव भिन्न *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    प्रत्यागम 0;
}
