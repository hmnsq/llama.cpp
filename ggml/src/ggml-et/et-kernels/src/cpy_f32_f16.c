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

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_cont_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env || !params) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        प्रत्यागम -1;
    }

    ध्रुव वर्ण * src_दत्त = (ध्रुव वर्ण *) src0->दत्त;
    uint16_t *   dst_दत्त = (uint16_t *) dst->दत्त;

    यदि (!src_दत्त || !dst_दत्त) {
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

    ध्रुव int64_t योग_elements = ne00 * ne01 * ne02 * ne03;

    यदि (योग_elements == 0) {
        प्रत्यागम 0;
    }

    // Check if src is contiguous F32
    ध्रुव bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ध्रुव int64_t elems_प्रत्येकम्_cl = 32;
    ध्रुव int64_t योग_cl     = (योग_elements + elems_प्रत्येकम्_cl - 1) / elems_प्रत्येकम्_cl;

    ध्रुव int64_t cl_प्रत्येकम्_thread = (योग_cl + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
    ध्रुव int64_t cl_start      = thread_id * cl_प्रत्येकम्_thread;
    int64_t       cl_अन्तः        = cl_start + cl_प्रत्येकम्_thread;
    यदि (cl_अन्तः > योग_cl) {
        cl_अन्तः = योग_cl;
    }
    यदि (cl_start >= योग_cl) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t es = cl_start * elems_प्रत्येकम्_cl;
    int64_t       ee = cl_अन्तः * elems_प्रत्येकम्_cl;
    यदि (ee > योग_elements) {
        ee = योग_elements;
    }

    यदि (src_contiguous) {
        // Fast path: src is contiguous F32
        ध्रुव भिन्न * src_f32 = (ध्रुव भिन्न *) src_दत्त;
        प्रति (int64_t i = es; i < ee; ++i) {
            dst_दत्त[i] = fp32_गन्तव्यम्_fp16(src_f32[i]);
        }
    } अन्यथा {
        // General path: stride-aware read
        प्रति (int64_t idx = es; idx < ee; ++idx) {
            ध्रुव int64_t i00  = idx % ne00;
            ध्रुव int64_t rem1 = idx / ne00;
            ध्रुव int64_t i01  = rem1 % ne01;
            ध्रुव int64_t rem2 = rem1 / ne01;
            ध्रुव int64_t i02  = rem2 % ne02;
            ध्रुव int64_t i03  = rem2 / ne02;

            ध्रुव भिन्न val = *(ध्रुव भिन्न *) (src_दत्त + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_दत्त[idx]   = fp32_गन्तव्यम्_fp16(val);
        }
    }

    प्रत्यागम 0;
}
