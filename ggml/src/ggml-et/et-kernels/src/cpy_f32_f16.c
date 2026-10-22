//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdbool.h>
#समावेश <stdint.h>

काष्ठा ggml_et_cont_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
};

पूर्णांक entry_point(काष्ठा ggml_et_cont_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env || !params) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        वापस -1;
    }

    स्थिर अक्षर * src_तथ्य = (स्थिर अक्षर *) src0->तथ्य;
    uint16_t *   dst_तथ्य = (uint16_t *) dst->तथ्य;

    अगर (!src_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];
    स्थिर int64_t ne03 = src0->ne[3];

    स्थिर int64_t nb00 = src0->nb[0];
    स्थिर int64_t nb01 = src0->nb[1];
    स्थिर int64_t nb02 = src0->nb[2];
    स्थिर int64_t nb03 = src0->nb[3];

    स्थिर int64_t योग_elements = ne00 * ne01 * ne02 * ne03;

    अगर (योग_elements == 0) {
        वापस 0;
    }

    // Check if src is contiguous F32
    स्थिर bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    स्थिर int64_t elems_per_cl = 32;
    स्थिर int64_t योग_cl     = (योग_elements + elems_per_cl - 1) / elems_per_cl;

    स्थिर int64_t cl_per_thread = (योग_cl + अंक_threads - 1) / अंक_threads;
    स्थिर int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    अगर (cl_end > योग_cl) {
        cl_end = योग_cl;
    }
    अगर (cl_start >= योग_cl) {
        वापस 0;
    }

    स्थिर int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    अगर (ee > योग_elements) {
        ee = योग_elements;
    }

    अगर (src_contiguous) {
        // Fast path: src is contiguous F32
        स्थिर भग्न * src_f32 = (स्थिर भग्न *) src_तथ्य;
        क्रम (int64_t i = es; i < ee; ++i) {
            dst_तथ्य[i] = fp32_to_fp16(src_f32[i]);
        }
    } अन्यथा {
        // General path: stride-aware read
        क्रम (int64_t idx = es; idx < ee; ++idx) {
            स्थिर int64_t i00  = idx % ne00;
            स्थिर int64_t rem1 = idx / ne00;
            स्थिर int64_t i01  = rem1 % ne01;
            स्थिर int64_t rem2 = rem1 / ne01;
            स्थिर int64_t i02  = rem2 % ne02;
            स्थिर int64_t i03  = rem2 / ne02;

            स्थिर भग्न val = *(स्थिर भग्न *) (src_तथ्य + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_तथ्य[idx]   = fp32_to_fp16(val);
        }
    }

    वापस 0;
}
