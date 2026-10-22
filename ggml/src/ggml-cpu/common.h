#pragma once

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "traits.h"
#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#अन्तर्भाव <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
स्थिर अन्तःस्थ ggml_fp16_t f32_to_f16(भिन्न x) {
    पटिनिवत्त GGML_CPU_FP32_TO_FP16(x);
}

स्थिर अन्तःस्थ भिन्न f16_to_f32(ggml_fp16_t x) {
    पटिनिवत्त GGML_CPU_FP16_TO_FP32(x);
}

स्थिर अन्तःस्थ ggml_bf16_t f32_to_bf16(भिन्न x) {
    पटिनिवत्त GGML_FP32_TO_BF16(x);
}

स्थिर अन्तःस्थ भिन्न bf16_to_f32(ggml_bf16_t x) {
    पटिनिवत्त GGML_BF16_TO_FP32(x);
}

स्थिर अन्तःस्थ भिन्न i32_to_f32(int32_t x) {
    पटिनिवत्त x;
}

स्थिर अन्तःस्थ int32_t f32_to_i32(भिन्न x) {
    पटिनिवत्त x;
}

स्थिर अन्तःस्थ भिन्न f32_to_f32(भिन्न x) {
    पटिनिवत्त x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
संरचना type_conversion_table;

template <>
संरचना type_conversion_table<ggml_fp16_t> {
    स्थिर constexpr भिन्न (*to_f32)(ggml_fp16_t) = f16_to_f32;
    स्थिर constexpr ggml_fp16_t (*from_f32)(भिन्न) = f32_to_f16;
};

template <>
संरचना type_conversion_table<भिन्न> {
    स्थिर constexpr भिन्न (*to_f32)(भिन्न) = f32_to_f32;
    स्थिर constexpr भिन्न (*from_f32)(भिन्न) = f32_to_f32;
};

template <>
संरचना type_conversion_table<ggml_bf16_t> {
    स्थिर constexpr भिन्न (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    स्थिर constexpr ggml_bf16_t (*from_f32)(भिन्न) = f32_to_bf16;
};

template <>
संरचना type_conversion_table<int32_t> {
    स्थिर constexpr भिन्न (*to_f32)(int32_t) = i32_to_f32;
    स्थिर constexpr int32_t (*from_f32)(भिन्न) = f32_to_i32;
};

स्थिर std::pair<int64_t, int64_t> get_thread_range(ध्रुव संरचना ggml_compute_params * params, ध्रुव संरचना ggml_tensor * src0) {
    ध्रुव int64_t ith = params->ith;
    ध्रुव int64_t nth = params->nth;

    ध्रुव int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    पटिनिवत्त {ir0, ir1};
}

संरचना ggml_fa_tile_config {
    स्थिर constexpr size_t Q  = GGML_FA_TILE_Q;
    स्थिर constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
