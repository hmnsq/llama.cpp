#pragma once

#شامل "ggml.h"
#شامل "traits.h"
#شامل "ggml-cpu-impl.h"
#شامل "ggml-impl.h"
#شامل "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#شامل <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
ایستا درونخطی ggml_fp16_t f32_to_f16(اعشاری x) {
    بازگشت GGML_CPU_FP32_TO_FP16(x);
}

ایستا درونخطی اعشاری f16_to_f32(ggml_fp16_t x) {
    بازگشت GGML_CPU_FP16_TO_FP32(x);
}

ایستا درونخطی ggml_bf16_t f32_to_bf16(اعشاری x) {
    بازگشت GGML_FP32_TO_BF16(x);
}

ایستا درونخطی اعشاری bf16_to_f32(ggml_bf16_t x) {
    بازگشت GGML_BF16_TO_FP32(x);
}

ایستا درونخطی اعشاری i32_to_f32(int32_t x) {
    بازگشت x;
}

ایستا درونخطی int32_t f32_to_i32(اعشاری x) {
    بازگشت x;
}

ایستا درونخطی اعشاری f32_to_f32(اعشاری x) {
    بازگشت x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
ساختار type_conversion_table;

template <>
ساختار type_conversion_table<ggml_fp16_t> {
    ایستا constexpr اعشاری (*to_f32)(ggml_fp16_t) = f16_to_f32;
    ایستا constexpr ggml_fp16_t (*from_f32)(اعشاری) = f32_to_f16;
};

template <>
ساختار type_conversion_table<اعشاری> {
    ایستا constexpr اعشاری (*to_f32)(اعشاری) = f32_to_f32;
    ایستا constexpr اعشاری (*from_f32)(اعشاری) = f32_to_f32;
};

template <>
ساختار type_conversion_table<ggml_bf16_t> {
    ایستا constexpr اعشاری (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    ایستا constexpr ggml_bf16_t (*from_f32)(اعشاری) = f32_to_bf16;
};

template <>
ساختار type_conversion_table<int32_t> {
    ایستا constexpr اعشاری (*to_f32)(int32_t) = i32_to_f32;
    ایستا constexpr int32_t (*from_f32)(اعشاری) = f32_to_i32;
};

ایستا std::pair<int64_t, int64_t> get_thread_range(ثابت ساختار ggml_compute_params * params, ثابت ساختار ggml_tensor * src0) {
    ثابت int64_t ith = params->ith;
    ثابت int64_t nth = params->nth;

    ثابت int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    بازگشت {ir0, ir1};
}

ساختار ggml_fa_tile_config {
    ایستا constexpr size_t Q  = GGML_FA_TILE_Q;
    ایستا constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
