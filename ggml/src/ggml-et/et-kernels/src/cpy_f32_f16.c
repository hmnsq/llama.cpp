//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdbool.h>
#ضمن <stdint.h>

بنية ggml_et_cont_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
};

صحيح entry_point(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ܐܢ (!kernel_env || !params) {
        ܦܢܐ -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    ܐܢ (thread_id < 0) {
        ܦܢܐ 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    ܐܢ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        ܦܢܐ -1;
    }

    ثابت حرف * src_data = (ثابت حرف *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    ܐܢ (!src_data || !dst_data) {
        ܦܢܐ -1;
    }

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت int64_t nb00 = src0->nb[0];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    ܐܢ (total_elements == 0) {
        ܦܢܐ 0;
    }

    // Check if src is contiguous F32
    ثابت bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ثابت int64_t elems_per_cl = 32;
    ثابت int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ثابت int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    ثابت int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ܐܢ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ܐܢ (cl_start >= total_cl) {
        ܦܢܐ 0;
    }

    ثابت int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ܐܢ (ee > total_elements) {
        ee = total_elements;
    }

    ܐܢ (src_contiguous) {
        // Fast path: src is contiguous F32
        ثابت عائم * src_f32 = (ثابت عائم *) src_data;
        لكل (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } ܐܚܪܢܐ {
        // General path: stride-aware read
        لكل (int64_t idx = es; idx < ee; ++idx) {
            ثابت int64_t i00  = idx % ne00;
            ثابت int64_t rem1 = idx / ne00;
            ثابت int64_t i01  = rem1 % ne01;
            ثابت int64_t rem2 = rem1 / ne01;
            ثابت int64_t i02  = rem2 % ne02;
            ثابت int64_t i03  = rem2 / ne02;

            ثابت عائم val = *(ثابت عائم *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    ܦܢܐ 0;
}
