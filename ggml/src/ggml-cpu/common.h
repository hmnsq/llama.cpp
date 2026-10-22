#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "traits.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "simd-mappings.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_FA_TILE_Q  64
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_FA_TILE_KV 64

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus

#ಅನ್ತರ್ಭಾವ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_fp16_t f32_ಗಮ್ಯ_f16(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ GGML_CPU_FP32_ಗಮ್ಯ_FP16(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ f16_ಗಮ್ಯ_f32(ggml_fp16_t x) {
    ಹಿಂತಿರುಗಿ GGML_CPU_FP16_ಗಮ್ಯ_FP32(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_bf16_t f32_ಗಮ್ಯ_bf16(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ GGML_FP32_ಗಮ್ಯ_BF16(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ bf16_ಗಮ್ಯ_f32(ggml_bf16_t x) {
    ಹಿಂತಿರುಗಿ GGML_BF16_ಗಮ್ಯ_FP32(x);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ i32_ಗಮ್ಯ_f32(int32_t x) {
    ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t f32_ಗಮ್ಯ_i32(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ f32_ಗಮ್ಯ_f32(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ x;
}

// TODO - merge this into the traits table, after using row-based conversions
ಮಾದರಿ <class T>
ರಚನೆ type_conversion_ಕೋಷ್ಟಕ;

ಮಾದರಿ <>
ರಚನೆ type_conversion_ಕೋಷ್ಟಕ<ggml_fp16_t> {
    ಸ್ಥಿರ constexpr ದಶಮಾಂಶ (*ಗಮ್ಯ_f32)(ggml_fp16_t) = f16_ಗಮ್ಯ_f32;
    ಸ್ಥಿರ constexpr ggml_fp16_t (*from_f32)(ದಶಮಾಂಶ) = f32_ಗಮ್ಯ_f16;
};

ಮಾದರಿ <>
ರಚನೆ type_conversion_ಕೋಷ್ಟಕ<ದಶಮಾಂಶ> {
    ಸ್ಥಿರ constexpr ದಶಮಾಂಶ (*ಗಮ್ಯ_f32)(ದಶಮಾಂಶ) = f32_ಗಮ್ಯ_f32;
    ಸ್ಥಿರ constexpr ದಶಮಾಂಶ (*from_f32)(ದಶಮಾಂಶ) = f32_ಗಮ್ಯ_f32;
};

ಮಾದರಿ <>
ರಚನೆ type_conversion_ಕೋಷ್ಟಕ<ggml_bf16_t> {
    ಸ್ಥಿರ constexpr ದಶಮಾಂಶ (*ಗಮ್ಯ_f32)(ggml_bf16_t) = bf16_ಗಮ್ಯ_f32;
    ಸ್ಥಿರ constexpr ggml_bf16_t (*from_f32)(ದಶಮಾಂಶ) = f32_ಗಮ್ಯ_bf16;
};

ಮಾದರಿ <>
ರಚನೆ type_conversion_ಕೋಷ್ಟಕ<int32_t> {
    ಸ್ಥಿರ constexpr ದಶಮಾಂಶ (*ಗಮ್ಯ_f32)(int32_t) = i32_ಗಮ್ಯ_f32;
    ಸ್ಥಿರ constexpr int32_t (*from_f32)(ದಶಮಾಂಶ) = f32_ಗಮ್ಯ_i32;
};

ಸ್ಥಿರ std::pair<int64_t, int64_t> get_thread_range(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0) {
    ಸ್ಥಿರಾಂಕ int64_t ith = params->ith;
    ಸ್ಥಿರಾಂಕ int64_t nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ಹಿಂತಿರುಗಿ {ir0, ir1};
}

ರಚನೆ ggml_fa_tile_config {
    ಸ್ಥಿರ constexpr size_t Q  = GGML_FA_TILE_Q;
    ಸ್ಥಿರ constexpr size_t KV = GGML_FA_TILE_KV;
};

#ಆದರೆ_ಅಂತ್ಯ
