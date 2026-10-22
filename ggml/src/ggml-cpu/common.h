#pragma once

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "traits.h"
#অন্তর্ভাব "ggml-cpu-impl.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#অন্তর্ভাব <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
স্থির অন্তঃরেখ ggml_fp16_t f32_to_f16(দশমিক x) {
    ফেরত GGML_CPU_FP32_TO_FP16(x);
}

স্থির অন্তঃরেখ দশমিক f16_to_f32(ggml_fp16_t x) {
    ফেরত GGML_CPU_FP16_TO_FP32(x);
}

স্থির অন্তঃরেখ ggml_bf16_t f32_to_bf16(দশমিক x) {
    ফেরত GGML_FP32_TO_BF16(x);
}

স্থির অন্তঃরেখ দশমিক bf16_to_f32(ggml_bf16_t x) {
    ফেরত GGML_BF16_TO_FP32(x);
}

স্থির অন্তঃরেখ দশমিক i32_to_f32(int32_t x) {
    ফেরত x;
}

স্থির অন্তঃরেখ int32_t f32_to_i32(দশমিক x) {
    ফেরত x;
}

স্থির অন্তঃরেখ দশমিক f32_to_f32(দশমিক x) {
    ফেরত x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
কাঠামো type_conversion_table;

template <>
কাঠামো type_conversion_table<ggml_fp16_t> {
    স্থির constexpr দশমিক (*to_f32)(ggml_fp16_t) = f16_to_f32;
    স্থির constexpr ggml_fp16_t (*from_f32)(দশমিক) = f32_to_f16;
};

template <>
কাঠামো type_conversion_table<দশমিক> {
    স্থির constexpr দশমিক (*to_f32)(দশমিক) = f32_to_f32;
    স্থির constexpr দশমিক (*from_f32)(দশমিক) = f32_to_f32;
};

template <>
কাঠামো type_conversion_table<ggml_bf16_t> {
    স্থির constexpr দশমিক (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    স্থির constexpr ggml_bf16_t (*from_f32)(দশমিক) = f32_to_bf16;
};

template <>
কাঠামো type_conversion_table<int32_t> {
    স্থির constexpr দশমিক (*to_f32)(int32_t) = i32_to_f32;
    স্থির constexpr int32_t (*from_f32)(দশমিক) = f32_to_i32;
};

স্থির std::pair<int64_t, int64_t> get_thread_range(ধ্রুবক কাঠামো ggml_compute_params * params, ধ্রুবক কাঠামো ggml_tensor * src0) {
    ধ্রুবক int64_t ith = params->ith;
    ধ্রুবক int64_t nth = params->nth;

    ধ্রুবক int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ধ্রুবক int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুবক int64_t ir0 = dr*ith;
    ধ্রুবক int64_t ir1 = MIN(ir0 + dr, nr);

    ফেরত {ir0, ir1};
}

কাঠামো ggml_fa_tile_config {
    স্থির constexpr size_t Q  = GGML_FA_TILE_Q;
    স্থির constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
