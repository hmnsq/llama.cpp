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

صحيح ثبت_ټکی(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        أرجع -1;
    }

    ثابت حرف * src_معلومات = (ثابت حرف *) src0->معلومات;
    uint16_t *   dst_معلومات = (uint16_t *) dst->معلومات;

    إذا (!src_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت int64_t nb00 = src0->nb[0];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t ټول_elements = ne00 * ne01 * ne02 * ne03;

    إذا (ټول_elements == 0) {
        أرجع 0;
    }

    // Check if src is contiguous F32
    ثابت bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ثابت int64_t elems_فی_cl = 32;
    ثابت int64_t ټول_cl     = (ټول_elements + elems_فی_cl - 1) / elems_فی_cl;

    ثابت int64_t cl_فی_thread = (ټول_cl + عدد_تارونه - 1) / عدد_تارونه;
    ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
    int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
    إذا (cl_پای_ټکی > ټول_cl) {
        cl_پای_ټکی = ټول_cl;
    }
    إذا (cl_پیل >= ټول_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_پیل * elems_فی_cl;
    int64_t       ee = cl_پای_ټکی * elems_فی_cl;
    إذا (ee > ټول_elements) {
        ee = ټول_elements;
    }

    إذا (src_contiguous) {
        // Fast path: src is contiguous F32
        ثابت عائم * src_f32 = (ثابت عائم *) src_معلومات;
        لكل (int64_t i = es; i < ee; ++i) {
            dst_معلومات[i] = fp32_منزل_fp16(src_f32[i]);
        }
    } وإلا {
        // General path: stride-aware read
        لكل (int64_t idx = es; idx < ee; ++idx) {
            ثابت int64_t i00  = idx % ne00;
            ثابت int64_t rem1 = idx / ne00;
            ثابت int64_t i01  = rem1 % ne01;
            ثابت int64_t rem2 = rem1 / ne01;
            ثابت int64_t i02  = rem2 % ne02;
            ثابت int64_t i03  = rem2 / ne02;

            ثابت عائم val = *(ثابت عائم *) (src_معلومات + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_معلومات[idx]   = fp32_منزل_fp16(val);
        }
    }

    أرجع 0;
}
