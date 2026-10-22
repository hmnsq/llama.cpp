//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_cont_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor dst;
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_cont_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env || !params) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        ফেরত -1;
    }

    ধ্রুবক অক্ষর * src_তথ্য = (ধ্রুবক অক্ষর *) src0->তথ্য;
    uint16_t *   dst_তথ্য = (uint16_t *) dst->তথ্য;

    যদি (!src_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t ne00 = src0->ne[0];
    ধ্রুবক int64_t ne01 = src0->ne[1];
    ধ্রুবক int64_t ne02 = src0->ne[2];
    ধ্রুবক int64_t ne03 = src0->ne[3];

    ধ্রুবক int64_t nb00 = src0->nb[0];
    ধ্রুবক int64_t nb01 = src0->nb[1];
    ধ্রুবক int64_t nb02 = src0->nb[2];
    ধ্রুবক int64_t nb03 = src0->nb[3];

    ধ্রুবক int64_t যোগ_elements = ne00 * ne01 * ne02 * ne03;

    যদি (যোগ_elements == 0) {
        ফেরত 0;
    }

    // Check if src is contiguous F32
    ধ্রুবক bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ধ্রুবক int64_t elems_per_cl = 32;
    ধ্রুবক int64_t যোগ_cl     = (যোগ_elements + elems_per_cl - 1) / elems_per_cl;

    ধ্রুবক int64_t cl_per_thread = (যোগ_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    যদি (cl_end > যোগ_cl) {
        cl_end = যোগ_cl;
    }
    যদি (cl_start >= যোগ_cl) {
        ফেরত 0;
    }

    ধ্রুবক int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    যদি (ee > যোগ_elements) {
        ee = যোগ_elements;
    }

    যদি (src_contiguous) {
        // Fast path: src is contiguous F32
        ধ্রুবক দশমিক * src_f32 = (ধ্রুবক দশমিক *) src_তথ্য;
        জন্য (int64_t i = es; i < ee; ++i) {
            dst_তথ্য[i] = fp32_to_fp16(src_f32[i]);
        }
    } অন্যথা {
        // General path: stride-aware read
        জন্য (int64_t idx = es; idx < ee; ++idx) {
            ধ্রুবক int64_t i00  = idx % ne00;
            ধ্রুবক int64_t rem1 = idx / ne00;
            ধ্রুবক int64_t i01  = rem1 % ne01;
            ধ্রুবক int64_t rem2 = rem1 / ne01;
            ধ্রুবক int64_t i02  = rem2 % ne02;
            ধ্রুবক int64_t i03  = rem2 / ne02;

            ধ্রুবক দশমিক val = *(ধ্রুবক দশমিক *) (src_তথ্য + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_তথ্য[idx]   = fp32_to_fp16(val);
        }
    }

    ফেরত 0;
}
