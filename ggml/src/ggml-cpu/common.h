#pragma once

#ضمن "ggml.h"
#ضمن "traits.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml-impl.h"
#ضمن "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#ضمن <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
ساكن مضمن ggml_fp16_t f32_to_f16(عائم x) {
    إرجاع GGML_CPU_FP32_TO_FP16(x);
}

ساكن مضمن عائم f16_to_f32(ggml_fp16_t x) {
    إرجاع GGML_CPU_FP16_TO_FP32(x);
}

ساكن مضمن ggml_bf16_t f32_to_bf16(عائم x) {
    إرجاع GGML_FP32_TO_BF16(x);
}

ساكن مضمن عائم bf16_to_f32(ggml_bf16_t x) {
    إرجاع GGML_BF16_TO_FP32(x);
}

ساكن مضمن عائم i32_to_f32(int32_t x) {
    إرجاع x;
}

ساكن مضمن int32_t f32_to_i32(عائم x) {
    إرجاع x;
}

ساكن مضمن عائم f32_to_f32(عائم x) {
    إرجاع x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
بنية type_conversion_table;

template <>
بنية type_conversion_table<ggml_fp16_t> {
    ساكن constexpr عائم (*to_f32)(ggml_fp16_t) = f16_to_f32;
    ساكن constexpr ggml_fp16_t (*from_f32)(عائم) = f32_to_f16;
};

template <>
بنية type_conversion_table<عائم> {
    ساكن constexpr عائم (*to_f32)(عائم) = f32_to_f32;
    ساكن constexpr عائم (*from_f32)(عائم) = f32_to_f32;
};

template <>
بنية type_conversion_table<ggml_bf16_t> {
    ساكن constexpr عائم (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    ساكن constexpr ggml_bf16_t (*from_f32)(عائم) = f32_to_bf16;
};

template <>
بنية type_conversion_table<int32_t> {
    ساكن constexpr عائم (*to_f32)(int32_t) = i32_to_f32;
    ساكن constexpr int32_t (*from_f32)(عائم) = f32_to_i32;
};

ساكن std::pair<int64_t, int64_t> get_thread_range(ثابت بنية ggml_compute_params * params, ثابت بنية ggml_tensor * src0) {
    ثابت int64_t ith = params->ith;
    ثابت int64_t nth = params->nth;

    ثابت int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    إرجاع {ir0, ir1};
}

بنية ggml_fa_tile_config {
    ساكن constexpr size_t Q  = GGML_FA_TILE_Q;
    ساكن constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
