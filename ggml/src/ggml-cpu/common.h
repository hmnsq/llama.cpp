#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_FA_TILE_Q  64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_FA_TILE_KV 64

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus

#ਅਨ੍ਤਰ੍ਭਾਵ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_fp16_t f32_ਮੰਜ਼ਿਲ_f16(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ f16_ਮੰਜ਼ਿਲ_f32(ggml_fp16_t x) {
    ਵਾਪਸ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_bf16_t f32_ਮੰਜ਼ਿਲ_bf16(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ GGML_FP32_ਮੰਜ਼ਿਲ_BF16(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ bf16_ਮੰਜ਼ਿਲ_f32(ggml_bf16_t x) {
    ਵਾਪਸ GGML_BF16_ਮੰਜ਼ਿਲ_FP32(x);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ i32_ਮੰਜ਼ਿਲ_f32(int32_t x) {
    ਵਾਪਸ x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t f32_ਮੰਜ਼ਿਲ_i32(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ f32_ਮੰਜ਼ਿਲ_f32(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ x;
}

// TODO - merge this into the traits table, after using row-based conversions
ਸਾਂਚਾ <class T>
ਬਣਤਰ type_conversion_ਸਾਰਣੀ;

ਸਾਂਚਾ <>
ਬਣਤਰ type_conversion_ਸਾਰਣੀ<ggml_fp16_t> {
    ਸਥਿਰ constexpr ਦਸ਼ਮਲਵ (*ਮੰਜ਼ਿਲ_f32)(ggml_fp16_t) = f16_ਮੰਜ਼ਿਲ_f32;
    ਸਥਿਰ constexpr ggml_fp16_t (*from_f32)(ਦਸ਼ਮਲਵ) = f32_ਮੰਜ਼ਿਲ_f16;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_conversion_ਸਾਰਣੀ<ਦਸ਼ਮਲਵ> {
    ਸਥਿਰ constexpr ਦਸ਼ਮਲਵ (*ਮੰਜ਼ਿਲ_f32)(ਦਸ਼ਮਲਵ) = f32_ਮੰਜ਼ਿਲ_f32;
    ਸਥਿਰ constexpr ਦਸ਼ਮਲਵ (*from_f32)(ਦਸ਼ਮਲਵ) = f32_ਮੰਜ਼ਿਲ_f32;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_conversion_ਸਾਰਣੀ<ggml_bf16_t> {
    ਸਥਿਰ constexpr ਦਸ਼ਮਲਵ (*ਮੰਜ਼ਿਲ_f32)(ggml_bf16_t) = bf16_ਮੰਜ਼ਿਲ_f32;
    ਸਥਿਰ constexpr ggml_bf16_t (*from_f32)(ਦਸ਼ਮਲਵ) = f32_ਮੰਜ਼ਿਲ_bf16;
};

ਸਾਂਚਾ <>
ਬਣਤਰ type_conversion_ਸਾਰਣੀ<int32_t> {
    ਸਥਿਰ constexpr ਦਸ਼ਮਲਵ (*ਮੰਜ਼ਿਲ_f32)(int32_t) = i32_ਮੰਜ਼ਿਲ_f32;
    ਸਥਿਰ constexpr int32_t (*from_f32)(ਦਸ਼ਮਲਵ) = f32_ਮੰਜ਼ਿਲ_i32;
};

ਸਥਿਰ std::pair<int64_t, int64_t> get_thread_range(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਅਚਲ ਬਣਤਰ ggml_tensor * src0) {
    ਅਚਲ int64_t ith = params->ith;
    ਅਚਲ int64_t nth = params->nth;

    ਅਚਲ int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਵਾਪਸ {ir0, ir1};
}

ਬਣਤਰ ggml_fa_tile_config {
    ਸਥਿਰ constexpr size_t Q  = GGML_FA_TILE_Q;
    ਸਥਿਰ constexpr size_t KV = GGML_FA_TILE_KV;
};

#ਜੇ_ਅੰਤ
