//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_cont_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
};

પૂર્ણાંક entry_point(સંરચના ggml_et_cont_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env || !params) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        પાછા -1;
    }

    અચલ અક્ષર * src_data = (અચલ અક્ષર *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    જો (!src_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];
    અચલ int64_t ne02 = src0->ne[2];
    અચલ int64_t ne03 = src0->ne[3];

    અચલ int64_t nb00 = src0->nb[0];
    અચલ int64_t nb01 = src0->nb[1];
    અચલ int64_t nb02 = src0->nb[2];
    અચલ int64_t nb03 = src0->nb[3];

    અચલ int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    જો (total_elements == 0) {
        પાછા 0;
    }

    // Check if src is contiguous F32
    અચલ bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    અચલ int64_t elems_per_cl = 32;
    અચલ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    અચલ int64_t cl_per_thread = (total_cl + અંક_threads - 1) / અંક_threads;
    અચલ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    જો (cl_end > total_cl) {
        cl_end = total_cl;
    }
    જો (cl_start >= total_cl) {
        પાછા 0;
    }

    અચલ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    જો (ee > total_elements) {
        ee = total_elements;
    }

    જો (src_contiguous) {
        // Fast path: src is contiguous F32
        અચલ દશાંશ * src_f32 = (અચલ દશાંશ *) src_data;
        માટે (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } નહીંતર {
        // General path: stride-aware read
        માટે (int64_t idx = es; idx < ee; ++idx) {
            અચલ int64_t i00  = idx % ne00;
            અચલ int64_t rem1 = idx / ne00;
            અચલ int64_t i01  = rem1 % ne01;
            અચલ int64_t rem2 = rem1 / ne01;
            અચલ int64_t i02  = rem2 % ne02;
            અચલ int64_t i03  = rem2 / ne02;

            અચલ દશાંશ val = *(અચલ દશાંશ *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    પાછા 0;
}
