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

صحيح اندراج_نقطہ(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        أرجع -1;
    }

    ثابت حرف * src_انکڑے = (ثابت حرف *) src0->انکڑے;
    uint16_t *   dst_انکڑے = (uint16_t *) dst->انکڑے;

    إذا (!src_انکڑے || !dst_انکڑے) {
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

    ثابت int64_t کل_elements = ne00 * ne01 * ne02 * ne03;

    إذا (کل_elements == 0) {
        أرجع 0;
    }

    // Check if src is contiguous F32
    ثابت bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ثابت int64_t elems_فی_cl = 32;
    ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

    ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_شروع      = thread_id * cl_فی_thread;
    int64_t       cl_سرا        = cl_شروع + cl_فی_thread;
    إذا (cl_سرا > کل_cl) {
        cl_سرا = کل_cl;
    }
    إذا (cl_شروع >= کل_cl) {
        أرجع 0;
    }

    ثابت int64_t es = cl_شروع * elems_فی_cl;
    int64_t       ee = cl_سرا * elems_فی_cl;
    إذا (ee > کل_elements) {
        ee = کل_elements;
    }

    إذا (src_contiguous) {
        // Fast path: src is contiguous F32
        ثابت عائم * src_f32 = (ثابت عائم *) src_انکڑے;
        لكل (int64_t i = es; i < ee; ++i) {
            dst_انکڑے[i] = fp32_منزل_fp16(src_f32[i]);
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

            ثابت عائم val = *(ثابت عائم *) (src_انکڑے + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_انکڑے[idx]   = fp32_منزل_fp16(val);
        }
    }

    أرجع 0;
}
