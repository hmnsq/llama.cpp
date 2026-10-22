//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_cont_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_cont_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env || !params) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ଅକ୍ଷର * src_data = (ଅଚଳ ଅକ୍ଷର *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    ଯଦି (!src_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t ne00 = src0->ne[0];
    ଅଚଳ int64_t ne01 = src0->ne[1];
    ଅଚଳ int64_t ne02 = src0->ne[2];
    ଅଚଳ int64_t ne03 = src0->ne[3];

    ଅଚଳ int64_t nb00 = src0->nb[0];
    ଅଚଳ int64_t nb01 = src0->nb[1];
    ଅଚଳ int64_t nb02 = src0->nb[2];
    ଅଚଳ int64_t nb03 = src0->nb[3];

    ଅଚଳ int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    ଯଦି (total_elements == 0) {
        ଫେରନ୍ତୁ 0;
    }

    // Check if src is contiguous F32
    ଅଚଳ bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ଅଚଳ int64_t elems_per_cl = 32;
    ଅଚଳ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ଅଚଳ int64_t cl_per_thread = (total_cl + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
    ଅଚଳ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ଯଦି (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ଯଦି (cl_start >= total_cl) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ଯଦି (ee > total_elements) {
        ee = total_elements;
    }

    ଯଦି (src_contiguous) {
        // Fast path: src is contiguous F32
        ଅଚଳ ଦଶମିକ * src_f32 = (ଅଚଳ ଦଶମିକ *) src_data;
        ପାଇଁ (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } ଅନ୍ୟଥା {
        // General path: stride-aware read
        ପାଇଁ (int64_t idx = es; idx < ee; ++idx) {
            ଅଚଳ int64_t i00  = idx % ne00;
            ଅଚଳ int64_t rem1 = idx / ne00;
            ଅଚଳ int64_t i01  = rem1 % ne01;
            ଅଚଳ int64_t rem2 = rem1 / ne01;
            ଅଚଳ int64_t i02  = rem2 % ne02;
            ଅଚଳ int64_t i03  = rem2 / ne02;

            ଅଚଳ ଦଶମିକ val = *(ଅଚଳ ଦଶମିକ *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    ଫେରନ୍ତୁ 0;
}
