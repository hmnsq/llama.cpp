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

पूर्णांक entry_point(संरचना ggml_et_cont_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env || !params) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        परत -1;
    }

    स्थिरांक अक्षर * src_data = (स्थिरांक अक्षर *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    जर (!src_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne01 = src0->ne[1];
    स्थिरांक int64_t ne02 = src0->ne[2];
    स्थिरांक int64_t ne03 = src0->ne[3];

    स्थिरांक int64_t nb00 = src0->nb[0];
    स्थिरांक int64_t nb01 = src0->nb[1];
    स्थिरांक int64_t nb02 = src0->nb[2];
    स्थिरांक int64_t nb03 = src0->nb[3];

    स्थिरांक int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    जर (total_elements == 0) {
        परत 0;
    }

    // Check if src is contiguous F32
    स्थिरांक bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    स्थिरांक int64_t elems_per_cl = 32;
    स्थिरांक int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    स्थिरांक int64_t cl_per_thread = (total_cl + अंक_threads - 1) / अंक_threads;
    स्थिरांक int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    जर (cl_end > total_cl) {
        cl_end = total_cl;
    }
    जर (cl_start >= total_cl) {
        परत 0;
    }

    स्थिरांक int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    जर (ee > total_elements) {
        ee = total_elements;
    }

    जर (src_contiguous) {
        // Fast path: src is contiguous F32
        स्थिरांक दशांश * src_f32 = (स्थिरांक दशांश *) src_data;
        साठी (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } अन्यथा {
        // General path: stride-aware read
        साठी (int64_t idx = es; idx < ee; ++idx) {
            स्थिरांक int64_t i00  = idx % ne00;
            स्थिरांक int64_t rem1 = idx / ne00;
            स्थिरांक int64_t i01  = rem1 % ne01;
            स्थिरांक int64_t rem2 = rem1 / ne01;
            स्थिरांक int64_t i02  = rem2 % ne02;
            स्थिरांक int64_t i03  = rem2 / ne02;

            स्थिरांक दशांश val = *(स्थिरांक दशांश *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    परत 0;
}
