//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_cont_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
};

முழு entry_point(கட்டமைப்பு ggml_et_cont_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env || !params) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        திரும்பு -1;
    }

    மாறிலி எழுத்து * src_data = (மாறிலி எழுத்து *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    எனில் (!src_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t ne00 = src0->ne[0];
    மாறிலி int64_t ne01 = src0->ne[1];
    மாறிலி int64_t ne02 = src0->ne[2];
    மாறிலி int64_t ne03 = src0->ne[3];

    மாறிலி int64_t nb00 = src0->nb[0];
    மாறிலி int64_t nb01 = src0->nb[1];
    மாறிலி int64_t nb02 = src0->nb[2];
    மாறிலி int64_t nb03 = src0->nb[3];

    மாறிலி int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    எனில் (total_elements == 0) {
        திரும்பு 0;
    }

    // Check if src is contiguous F32
    மாறிலி bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    மாறிலி int64_t elems_per_cl = 32;
    மாறிலி int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    மாறிலி int64_t cl_per_thread = (total_cl + இலக்கம்_threads - 1) / இலக்கம்_threads;
    மாறிலி int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    எனில் (cl_end > total_cl) {
        cl_end = total_cl;
    }
    எனில் (cl_start >= total_cl) {
        திரும்பு 0;
    }

    மாறிலி int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    எனில் (ee > total_elements) {
        ee = total_elements;
    }

    எனில் (src_contiguous) {
        // Fast path: src is contiguous F32
        மாறிலி தசம * src_f32 = (மாறிலி தசம *) src_data;
        க்கு (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } இல்லையெனில் {
        // General path: stride-aware read
        க்கு (int64_t idx = es; idx < ee; ++idx) {
            மாறிலி int64_t i00  = idx % ne00;
            மாறிலி int64_t rem1 = idx / ne00;
            மாறிலி int64_t i01  = rem1 % ne01;
            மாறிலி int64_t rem2 = rem1 / ne01;
            மாறிலி int64_t i02  = rem2 % ne02;
            மாறிலி int64_t i03  = rem2 / ne02;

            மாறிலி தசம val = *(மாறிலி தசம *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    திரும்பு 0;
}
