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

صحيح داخلا_نقطو(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

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

    ثابت int64_t ڪل_elements = ne00 * ne01 * ne02 * ne03;

    إذا (ڪل_elements == 0) {
        أرجع 0;
    }

    // Check if src is contiguous F32
    ثابت bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ثابت int64_t elems_في_cl = 32;
    ثابت int64_t ڪل_cl     = (ڪل_elements + elems_في_cl - 1) / elems_في_cl;

    ثابت int64_t cl_في_thread = (ڪل_cl + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
    ثابت int64_t cl_شروعات      = thread_id * cl_في_thread;
    int64_t       cl_پڇاڙي        = cl_شروعات + cl_في_thread;
    إذا (cl_پڇاڙي > ڪل_cl) {
        cl_پڇاڙي = ڪل_cl;
    }
    إذا (cl_شروعات >= ڪل_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_شروعات * elems_في_cl;
    int64_t       ee = cl_پڇاڙي * elems_في_cl;
    إذا (ee > ڪل_elements) {
        ee = ڪل_elements;
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
