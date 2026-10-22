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

ساخت ggml_et_cont_params {
    ساخت ggml_tensor src0;
    ساخت ggml_tensor dst;
};

صحیح اندراج_نقطہ(ساخت ggml_et_cont_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || !params) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        واپس -1;
    }

    ثابت حرف * src_ڈیٹا = (ثابت حرف *) src0->ڈیٹا;
    uint16_t *   dst_ڈیٹا = (uint16_t *) dst->ڈیٹا;

    اگر (!src_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
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

    اگر (کل_elements == 0) {
        واپس 0;
    }

    // Check if src is contiguous F32
    ثابت bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    ثابت int64_t elems_فی_cl = 32;
    ثابت int64_t کل_cl     = (کل_elements + elems_فی_cl - 1) / elems_فی_cl;

    ثابت int64_t cl_فی_thread = (کل_cl + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
    int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
    اگر (cl_انتہا > کل_cl) {
        cl_انتہا = کل_cl;
    }
    اگر (cl_آغاز >= کل_cl) {
        واپس 0;
    }

    ثابت int64_t es = cl_آغاز * elems_فی_cl;
    int64_t       ee = cl_انتہا * elems_فی_cl;
    اگر (ee > کل_elements) {
        ee = کل_elements;
    }

    اگر (src_contiguous) {
        // Fast path: src is contiguous F32
        ثابت اعشاری * src_f32 = (ثابت اعشاری *) src_ڈیٹا;
        برائے (int64_t i = es; i < ee; ++i) {
            dst_ڈیٹا[i] = fp32_مقصد_fp16(src_f32[i]);
        }
    } ورنہ {
        // General path: stride-aware read
        برائے (int64_t idx = es; idx < ee; ++idx) {
            ثابت int64_t i00  = idx % ne00;
            ثابت int64_t rem1 = idx / ne00;
            ثابت int64_t i01  = rem1 % ne01;
            ثابت int64_t rem2 = rem1 / ne01;
            ثابت int64_t i02  = rem2 % ne02;
            ثابت int64_t i03  = rem2 / ne02;

            ثابت اعشاری val = *(ثابت اعشاری *) (src_ڈیٹا + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_ڈیٹا[idx]   = fp32_مقصد_fp16(val);
        }
    }

    واپس 0;
}
