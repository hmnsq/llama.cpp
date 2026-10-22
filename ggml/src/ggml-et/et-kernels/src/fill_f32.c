//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_fill_params {
    संरचना ggml_tensor dst;  // F32 output tensor (contiguous)
    दशांश              c;    // Constant value to fill
};

पूर्णांक entry_point(संरचना ggml_et_fill_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * dst = &params->dst;

    जर (dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    दशांश * dst_data = (दशांश *) dst->data;
    जर (!dst_data) {
        परत -1;
    }

    स्थिरांक int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    जर (total_elements == 0) {
        परत 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    स्थिरांक int64_t elems_per_cl  = 16;
    स्थिरांक int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
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

    // Broadcast constant to all SIMD lanes
    दशांश c = params->c;
    __asm__ अस्थिर("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    स्थिरांक int64_t vec_end = es + ((ee - es) / 8) * 8;
    साठी (; i < vec_end; i += 8) {
        __asm__ अस्थिर("fsw.ps f10, %[d]\n" : [d] "=m"(*(दशांश (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    साठी (; i < ee; i++) {
        dst_data[i] = c;
    }

    परत 0;
}
