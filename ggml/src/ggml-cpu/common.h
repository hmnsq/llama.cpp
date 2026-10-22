#pragma once

#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "traits.h"
#அந்தர்பாவ "ggml-cpu-impl.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#அந்தர்பாவ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
நிலையான உள்வரி ggml_fp16_t f32_to_f16(தசம x) {
    திரும்பு GGML_CPU_FP32_TO_FP16(x);
}

நிலையான உள்வரி தசம f16_to_f32(ggml_fp16_t x) {
    திரும்பு GGML_CPU_FP16_TO_FP32(x);
}

நிலையான உள்வரி ggml_bf16_t f32_to_bf16(தசம x) {
    திரும்பு GGML_FP32_TO_BF16(x);
}

நிலையான உள்வரி தசம bf16_to_f32(ggml_bf16_t x) {
    திரும்பு GGML_BF16_TO_FP32(x);
}

நிலையான உள்வரி தசம i32_to_f32(int32_t x) {
    திரும்பு x;
}

நிலையான உள்வரி int32_t f32_to_i32(தசம x) {
    திரும்பு x;
}

நிலையான உள்வரி தசம f32_to_f32(தசம x) {
    திரும்பு x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
கட்டமைப்பு type_conversion_table;

template <>
கட்டமைப்பு type_conversion_table<ggml_fp16_t> {
    நிலையான constexpr தசம (*to_f32)(ggml_fp16_t) = f16_to_f32;
    நிலையான constexpr ggml_fp16_t (*from_f32)(தசம) = f32_to_f16;
};

template <>
கட்டமைப்பு type_conversion_table<தசம> {
    நிலையான constexpr தசம (*to_f32)(தசம) = f32_to_f32;
    நிலையான constexpr தசம (*from_f32)(தசம) = f32_to_f32;
};

template <>
கட்டமைப்பு type_conversion_table<ggml_bf16_t> {
    நிலையான constexpr தசம (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    நிலையான constexpr ggml_bf16_t (*from_f32)(தசம) = f32_to_bf16;
};

template <>
கட்டமைப்பு type_conversion_table<int32_t> {
    நிலையான constexpr தசம (*to_f32)(int32_t) = i32_to_f32;
    நிலையான constexpr int32_t (*from_f32)(தசம) = f32_to_i32;
};

நிலையான std::pair<int64_t, int64_t> get_thread_range(மாறிலி கட்டமைப்பு ggml_compute_params * params, மாறிலி கட்டமைப்பு ggml_tensor * src0) {
    மாறிலி int64_t ith = params->ith;
    மாறிலி int64_t nth = params->nth;

    மாறிலி int64_t nr  = ggml_nrows(src0);

    // rows per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    திரும்பு {ir0, ir1};
}

கட்டமைப்பு ggml_fa_tile_config {
    நிலையான constexpr size_t Q  = GGML_FA_TILE_Q;
    நிலையான constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
