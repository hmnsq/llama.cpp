#ہدایت once

#شامل "ggml.h"
#شامل "traits.h"
#شامل "ggml-cpu-impl.h"
#شامل "ggml-impl.h"
#شامل "simd-mappings.h"

#تعریف_کرو GGML_FA_TILE_Q  64
#تعریف_کرو GGML_FA_TILE_KV 64

#اگر_تعریف_شدہ __cplusplus

#شامل <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
ساکن اندرونی ggml_fp16_t f32_مقصد_f16(اعشاری x) {
    واپس GGML_CPU_FP32_مقصد_FP16(x);
}

ساکن اندرونی اعشاری f16_مقصد_f32(ggml_fp16_t x) {
    واپس GGML_CPU_FP16_مقصد_FP32(x);
}

ساکن اندرونی ggml_bf16_t f32_مقصد_bf16(اعشاری x) {
    واپس GGML_FP32_مقصد_BF16(x);
}

ساکن اندرونی اعشاری bf16_مقصد_f32(ggml_bf16_t x) {
    واپس GGML_BF16_مقصد_FP32(x);
}

ساکن اندرونی اعشاری i32_مقصد_f32(int32_t x) {
    واپس x;
}

ساکن اندرونی int32_t f32_مقصد_i32(اعشاری x) {
    واپس x;
}

ساکن اندرونی اعشاری f32_مقصد_f32(اعشاری x) {
    واپس x;
}

// TODO - merge this into the traits table, after using row-based conversions
سانچہ <class T>
ساخت type_conversion_جدول;

سانچہ <>
ساخت type_conversion_جدول<ggml_fp16_t> {
    ساکن constexpr اعشاری (*مقصد_f32)(ggml_fp16_t) = f16_مقصد_f32;
    ساکن constexpr ggml_fp16_t (*from_f32)(اعشاری) = f32_مقصد_f16;
};

سانچہ <>
ساخت type_conversion_جدول<اعشاری> {
    ساکن constexpr اعشاری (*مقصد_f32)(اعشاری) = f32_مقصد_f32;
    ساکن constexpr اعشاری (*from_f32)(اعشاری) = f32_مقصد_f32;
};

سانچہ <>
ساخت type_conversion_جدول<ggml_bf16_t> {
    ساکن constexpr اعشاری (*مقصد_f32)(ggml_bf16_t) = bf16_مقصد_f32;
    ساکن constexpr ggml_bf16_t (*from_f32)(اعشاری) = f32_مقصد_bf16;
};

سانچہ <>
ساخت type_conversion_جدول<int32_t> {
    ساکن constexpr اعشاری (*مقصد_f32)(int32_t) = i32_مقصد_f32;
    ساکن constexpr int32_t (*from_f32)(اعشاری) = f32_مقصد_i32;
};

ساکن std::pair<int64_t, int64_t> get_thread_range(ثابت ساخت ggml_compute_params * params, ثابت ساخت ggml_tensor * src0) {
    ثابت int64_t ith = params->ith;
    ثابت int64_t nth = params->nth;

    ثابت int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    واپس {ir0, ir1};
}

ساخت ggml_fa_tile_config {
    ساکن constexpr size_t Q  = GGML_FA_TILE_Q;
    ساکن constexpr size_t KV = GGML_FA_TILE_KV;
};

#اگر_ختم
