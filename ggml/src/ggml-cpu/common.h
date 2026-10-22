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
स्थिर ओळीत ggml_fp16_t f32_to_f16(दशांश x) {
    परत GGML_CPU_FP32_TO_FP16(x);
}

स्थिर ओळीत दशांश f16_to_f32(ggml_fp16_t x) {
    परत GGML_CPU_FP16_TO_FP32(x);
}

स्थिर ओळीत ggml_bf16_t f32_to_bf16(दशांश x) {
    परत GGML_FP32_TO_BF16(x);
}

स्थिर ओळीत दशांश bf16_to_f32(ggml_bf16_t x) {
    परत GGML_BF16_TO_FP32(x);
}

स्थिर ओळीत दशांश i32_to_f32(int32_t x) {
    परत x;
}

स्थिर ओळीत int32_t f32_to_i32(दशांश x) {
    परत x;
}

स्थिर ओळीत दशांश f32_to_f32(दशांश x) {
    परत x;
}

// TODO - merge this into the traits table, after using row-based conversions
ढाँचा <class T>
संरचना type_conversion_table;

ढाँचा <>
संरचना type_conversion_table<ggml_fp16_t> {
    स्थिर constexpr दशांश (*to_f32)(ggml_fp16_t) = f16_to_f32;
    स्थिर constexpr ggml_fp16_t (*from_f32)(दशांश) = f32_to_f16;
};

ढाँचा <>
संरचना type_conversion_table<दशांश> {
    स्थिर constexpr दशांश (*to_f32)(दशांश) = f32_to_f32;
    स्थिर constexpr दशांश (*from_f32)(दशांश) = f32_to_f32;
};

ढाँचा <>
संरचना type_conversion_table<ggml_bf16_t> {
    स्थिर constexpr दशांश (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    स्थिर constexpr ggml_bf16_t (*from_f32)(दशांश) = f32_to_bf16;
};

ढाँचा <>
संरचना type_conversion_table<int32_t> {
    स्थिर constexpr दशांश (*to_f32)(int32_t) = i32_to_f32;
    स्थिर constexpr int32_t (*from_f32)(दशांश) = f32_to_i32;
};

स्थिर std::pair<int64_t, int64_t> get_thread_range(स्थिरांक संरचना ggml_compute_params * params, स्थिरांक संरचना ggml_tensor * src0) {
    स्थिरांक int64_t ith = params->ith;
    स्थिरांक int64_t nth = params->nth;

    स्थिरांक int64_t nr  = ggml_nrows(src0);

    // rows per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    परत {ir0, ir1};
}

संरचना ggml_fa_tile_config {
    स्थिर constexpr माप_प्रकार Q  = GGML_FA_TILE_Q;
    स्थिर constexpr माप_प्रकार KV = GGML_FA_TILE_KV;
};

#endif
