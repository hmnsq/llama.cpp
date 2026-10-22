//******************************************************************************
// CPY F32 -> F16 Kernel
// Copies F32 source tensor to F16 destination tensor (contiguous output).
// Source may have arbitrary strides; destination must be contiguous.
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_cont_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor dst;
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_cont_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env || !params) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        തിരികെ -1;
    }

    സ്ഥിരം അക്ഷരം * src_data = (സ്ഥിരം അക്ഷരം *) src0->data;
    uint16_t *   dst_data = (uint16_t *) dst->data;

    എങ്കിൽ (!src_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne01 = src0->ne[1];
    സ്ഥിരം int64_t ne02 = src0->ne[2];
    സ്ഥിരം int64_t ne03 = src0->ne[3];

    സ്ഥിരം int64_t nb00 = src0->nb[0];
    സ്ഥിരം int64_t nb01 = src0->nb[1];
    സ്ഥിരം int64_t nb02 = src0->nb[2];
    സ്ഥിരം int64_t nb03 = src0->nb[3];

    സ്ഥിരം int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    എങ്കിൽ (total_elements == 0) {
        തിരികെ 0;
    }

    // Check if src is contiguous F32
    സ്ഥിരം bool src_contiguous =
        (nb00 == 4 && nb01 == ne00 * 4 && nb02 == ne00 * ne01 * 4 && nb03 == ne00 * ne01 * ne02 * 4);

    // Distribute by cache lines (16 F16 elements = 32 bytes = half cache line)
    // Use 32 elements per chunk to keep output cache-line aligned
    സ്ഥിരം int64_t elems_per_cl = 32;
    സ്ഥിരം int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    സ്ഥിരം int64_t cl_per_thread = (total_cl + അക്കം_threads - 1) / അക്കം_threads;
    സ്ഥിരം int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    എങ്കിൽ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    എങ്കിൽ (cl_start >= total_cl) {
        തിരികെ 0;
    }

    സ്ഥിരം int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    എങ്കിൽ (ee > total_elements) {
        ee = total_elements;
    }

    എങ്കിൽ (src_contiguous) {
        // Fast path: src is contiguous F32
        സ്ഥിരം ദശാംശം * src_f32 = (സ്ഥിരം ദശാംശം *) src_data;
        വേണ്ടി (int64_t i = es; i < ee; ++i) {
            dst_data[i] = fp32_to_fp16(src_f32[i]);
        }
    } അല്ലെങ്കിൽ {
        // General path: stride-aware read
        വേണ്ടി (int64_t idx = es; idx < ee; ++idx) {
            സ്ഥിരം int64_t i00  = idx % ne00;
            സ്ഥിരം int64_t rem1 = idx / ne00;
            സ്ഥിരം int64_t i01  = rem1 % ne01;
            സ്ഥിരം int64_t rem2 = rem1 / ne01;
            സ്ഥിരം int64_t i02  = rem2 % ne02;
            സ്ഥിരം int64_t i03  = rem2 / ne02;

            സ്ഥിരം ദശാംശം val = *(സ്ഥിരം ദശാംശം *) (src_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx]   = fp32_to_fp16(val);
        }
    }

    തിരികെ 0;
}
