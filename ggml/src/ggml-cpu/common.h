#pragma once

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "traits.h"
#അന്തര്ഭാവ "ggml-cpu-impl.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#അന്തര്ഭാവ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
സ്ഥിതം വരിയിൽ ggml_fp16_t f32_to_f16(ദശാംശം x) {
    തിരികെ GGML_CPU_FP32_TO_FP16(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം f16_to_f32(ggml_fp16_t x) {
    തിരികെ GGML_CPU_FP16_TO_FP32(x);
}

സ്ഥിതം വരിയിൽ ggml_bf16_t f32_to_bf16(ദശാംശം x) {
    തിരികെ GGML_FP32_TO_BF16(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം bf16_to_f32(ggml_bf16_t x) {
    തിരികെ GGML_BF16_TO_FP32(x);
}

സ്ഥിതം വരിയിൽ ദശാംശം i32_to_f32(int32_t x) {
    തിരികെ x;
}

സ്ഥിതം വരിയിൽ int32_t f32_to_i32(ദശാംശം x) {
    തിരികെ x;
}

സ്ഥിതം വരിയിൽ ദശാംശം f32_to_f32(ദശാംശം x) {
    തിരികെ x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
ഘടന type_conversion_table;

template <>
ഘടന type_conversion_table<ggml_fp16_t> {
    സ്ഥിതം constexpr ദശാംശം (*to_f32)(ggml_fp16_t) = f16_to_f32;
    സ്ഥിതം constexpr ggml_fp16_t (*from_f32)(ദശാംശം) = f32_to_f16;
};

template <>
ഘടന type_conversion_table<ദശാംശം> {
    സ്ഥിതം constexpr ദശാംശം (*to_f32)(ദശാംശം) = f32_to_f32;
    സ്ഥിതം constexpr ദശാംശം (*from_f32)(ദശാംശം) = f32_to_f32;
};

template <>
ഘടന type_conversion_table<ggml_bf16_t> {
    സ്ഥിതം constexpr ദശാംശം (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    സ്ഥിതം constexpr ggml_bf16_t (*from_f32)(ദശാംശം) = f32_to_bf16;
};

template <>
ഘടന type_conversion_table<int32_t> {
    സ്ഥിതം constexpr ദശാംശം (*to_f32)(int32_t) = i32_to_f32;
    സ്ഥിതം constexpr int32_t (*from_f32)(ദശാംശം) = f32_to_i32;
};

സ്ഥിതം std::pair<int64_t, int64_t> get_thread_range(സ്ഥിരം ഘടന ggml_compute_params * params, സ്ഥിരം ഘടന ggml_tensor * src0) {
    സ്ഥിരം int64_t ith = params->ith;
    സ്ഥിരം int64_t nth = params->nth;

    സ്ഥിരം int64_t nr  = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    തിരികെ {ir0, ir1};
}

ഘടന ggml_fa_tile_config {
    സ്ഥിതം constexpr size_t Q  = GGML_FA_TILE_Q;
    സ്ഥിതം constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
