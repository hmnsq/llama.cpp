#pragma once

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "traits.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#ಅನ್ತರ್ಭಾವ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_fp16_t f32_to_f16(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ GGML_CPU_FP32_TO_FP16(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ f16_to_f32(ggml_fp16_t x) {
    ಪ್ರತ್ಯಾಗಮ GGML_CPU_FP16_TO_FP32(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_bf16_t f32_to_bf16(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ GGML_FP32_TO_BF16(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ bf16_to_f32(ggml_bf16_t x) {
    ಪ್ರತ್ಯಾಗಮ GGML_BF16_TO_FP32(x);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ i32_to_f32(int32_t x) {
    ಪ್ರತ್ಯಾಗಮ x;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int32_t f32_to_i32(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ x;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ f32_to_f32(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
ಸಂರಚನಾ type_conversion_table;

template <>
ಸಂರಚನಾ type_conversion_table<ggml_fp16_t> {
    ಸ್ಥಿರ constexpr ಭಿನ್ನ (*to_f32)(ggml_fp16_t) = f16_to_f32;
    ಸ್ಥಿರ constexpr ggml_fp16_t (*from_f32)(ಭಿನ್ನ) = f32_to_f16;
};

template <>
ಸಂರಚನಾ type_conversion_table<ಭಿನ್ನ> {
    ಸ್ಥಿರ constexpr ಭಿನ್ನ (*to_f32)(ಭಿನ್ನ) = f32_to_f32;
    ಸ್ಥಿರ constexpr ಭಿನ್ನ (*from_f32)(ಭಿನ್ನ) = f32_to_f32;
};

template <>
ಸಂರಚನಾ type_conversion_table<ggml_bf16_t> {
    ಸ್ಥಿರ constexpr ಭಿನ್ನ (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    ಸ್ಥಿರ constexpr ggml_bf16_t (*from_f32)(ಭಿನ್ನ) = f32_to_bf16;
};

template <>
ಸಂರಚನಾ type_conversion_table<int32_t> {
    ಸ್ಥಿರ constexpr ಭಿನ್ನ (*to_f32)(int32_t) = i32_to_f32;
    ಸ್ಥಿರ constexpr int32_t (*from_f32)(ಭಿನ್ನ) = f32_to_i32;
};

ಸ್ಥಿರ std::pair<int64_t, int64_t> get_thread_range(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0) {
    ಧ್ರುವ int64_t ith = params->ith;
    ಧ್ರುವ int64_t nth = params->nth;

    ಧ್ರುವ int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ಧ್ರುವ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಧ್ರುವ int64_t ir0 = dr*ith;
    ಧ್ರುವ int64_t ir1 = MIN(ir0 + dr, nr);

    ಪ್ರತ್ಯಾಗಮ {ir0, ir1};
}

ಸಂರಚನಾ ggml_fa_tile_config {
    ಸ್ಥಿರ constexpr size_t Q  = GGML_FA_TILE_Q;
    ಸ್ಥಿರ constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
