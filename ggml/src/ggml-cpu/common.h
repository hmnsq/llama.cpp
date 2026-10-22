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
স্থির অন্তঃস্থ ggml_fp16_t f32_to_f16(ভিন্ন x) {
    প্রত্যাগম GGML_CPU_FP32_TO_FP16(x);
}

স্থির অন্তঃস্থ ভিন্ন f16_to_f32(ggml_fp16_t x) {
    প্রত্যাগম GGML_CPU_FP16_TO_FP32(x);
}

স্থির অন্তঃস্থ ggml_bf16_t f32_to_bf16(ভিন্ন x) {
    প্রত্যাগম GGML_FP32_TO_BF16(x);
}

স্থির অন্তঃস্থ ভিন্ন bf16_to_f32(ggml_bf16_t x) {
    প্রত্যাগম GGML_BF16_TO_FP32(x);
}

স্থির অন্তঃস্থ ভিন্ন i32_to_f32(int32_t x) {
    প্রত্যাগম x;
}

স্থির অন্তঃস্থ int32_t f32_to_i32(ভিন্ন x) {
    প্রত্যাগম x;
}

স্থির অন্তঃস্থ ভিন্ন f32_to_f32(ভিন্ন x) {
    প্রত্যাগম x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
সংরচনা type_conversion_table;

template <>
সংরচনা type_conversion_table<ggml_fp16_t> {
    স্থির constexpr ভিন্ন (*to_f32)(ggml_fp16_t) = f16_to_f32;
    স্থির constexpr ggml_fp16_t (*from_f32)(ভিন্ন) = f32_to_f16;
};

template <>
সংরচনা type_conversion_table<ভিন্ন> {
    স্থির constexpr ভিন্ন (*to_f32)(ভিন্ন) = f32_to_f32;
    স্থির constexpr ভিন্ন (*from_f32)(ভিন্ন) = f32_to_f32;
};

template <>
সংরচনা type_conversion_table<ggml_bf16_t> {
    স্থির constexpr ভিন্ন (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    স্থির constexpr ggml_bf16_t (*from_f32)(ভিন্ন) = f32_to_bf16;
};

template <>
সংরচনা type_conversion_table<int32_t> {
    স্থির constexpr ভিন্ন (*to_f32)(int32_t) = i32_to_f32;
    স্থির constexpr int32_t (*from_f32)(ভিন্ন) = f32_to_i32;
};

স্থির std::pair<int64_t, int64_t> get_thread_range(ধ্রুব সংরচনা ggml_compute_params * params, ধ্রুব সংরচনা ggml_tensor * src0) {
    ধ্রুব int64_t ith = params->ith;
    ধ্রুব int64_t nth = params->nth;

    ধ্রুব int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    প্রত্যাগম {ir0, ir1};
}

সংরচনা ggml_fa_tile_config {
    স্থির constexpr size_t Q  = GGML_FA_TILE_Q;
    স্থির constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
