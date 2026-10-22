//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdbool.h>
#شامل <stdint.h>

ساختار ggml_et_cont_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor dst;
};

صحیح entry_point(ساختار ggml_et_cont_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || !params) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        بازگشت -1;
    }

    ثابت نویسه * src_data = (ثابت نویسه *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    اگر (!src_data || !dst_data) {
        بازگشت -1;
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

    اگر (total_elements == 0) {
        بازگشت 0;
    }

    // Check if src is contiguous F32
    ثابت bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ثابت int64_t elems_per_cl = 32;
    ثابت int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ثابت int64_t cl_per_thread = (total_cl + رقم_threads - 1) / رقم_threads;
    ثابت int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    اگر (cl_end > total_cl) {
        cl_end = total_cl;
    }
    اگر (cl_start >= total_cl) {
        بازگشت 0;
    }

    ثابت int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    اگر (ee > total_elements) {
        ee = total_elements;
    }

    اگر (src_contiguous) {
        // Fast path: src is contiguous F32
        ثابت اعشاری * src_f32 = (ثابت اعشاری *) src_data;
        برای (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } وگرنه {
        // General path: stride-aware read
        برای (int64_t idx = es; idx < ee; ++idx) {
            ثابت int64_t i00  = idx % ne00;
            ثابت int64_t rem1 = idx / ne00;
            ثابت int64_t i01  = rem1 % ne01;
            ثابت int64_t rem2 = rem1 / ne01;
            ثابت int64_t i02  = rem2 % ne02;
            ثابت int64_t i03  = rem2 / ne02;

            ثابت اعشاری val = *(ثابت اعشاری *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    بازگشت 0;
}
