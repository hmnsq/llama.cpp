#निर्देशः once

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "traits.h"
#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "simd-mappings.h"

#परिभाषय GGML_FA_TILE_Q  64
#परिभाषय GGML_FA_TILE_KV 64

#यदि_परिभाषितम् __cplusplus

#अन्तर्भाव <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
स्थिर अन्तःस्थ ggml_fp16_t f32_गन्तव्यम्_f16(भिन्न x) {
    प्रत्यागम GGML_CPU_FP32_गन्तव्यम्_FP16(x);
}

स्थिर अन्तःस्थ भिन्न f16_गन्तव्यम्_f32(ggml_fp16_t x) {
    प्रत्यागम GGML_CPU_FP16_गन्तव्यम्_FP32(x);
}

स्थिर अन्तःस्थ ggml_bf16_t f32_गन्तव्यम्_bf16(भिन्न x) {
    प्रत्यागम GGML_FP32_गन्तव्यम्_BF16(x);
}

स्थिर अन्तःस्थ भिन्न bf16_गन्तव्यम्_f32(ggml_bf16_t x) {
    प्रत्यागम GGML_BF16_गन्तव्यम्_FP32(x);
}

स्थिर अन्तःस्थ भिन्न i32_गन्तव्यम्_f32(int32_t x) {
    प्रत्यागम x;
}

स्थिर अन्तःस्थ int32_t f32_गन्तव्यम्_i32(भिन्न x) {
    प्रत्यागम x;
}

स्थिर अन्तःस्थ भिन्न f32_गन्तव्यम्_f32(भिन्न x) {
    प्रत्यागम x;
}

// TODO - merge this into the traits table, after using row-based conversions
प्रतिरूपम् <class T>
संरचना type_conversion_सारणी;

प्रतिरूपम् <>
संरचना type_conversion_सारणी<ggml_fp16_t> {
    स्थिर constexpr भिन्न (*गन्तव्यम्_f32)(ggml_fp16_t) = f16_गन्तव्यम्_f32;
    स्थिर constexpr ggml_fp16_t (*from_f32)(भिन्न) = f32_गन्तव्यम्_f16;
};

प्रतिरूपम् <>
संरचना type_conversion_सारणी<भिन्न> {
    स्थिर constexpr भिन्न (*गन्तव्यम्_f32)(भिन्न) = f32_गन्तव्यम्_f32;
    स्थिर constexpr भिन्न (*from_f32)(भिन्न) = f32_गन्तव्यम्_f32;
};

प्रतिरूपम् <>
संरचना type_conversion_सारणी<ggml_bf16_t> {
    स्थिर constexpr भिन्न (*गन्तव्यम्_f32)(ggml_bf16_t) = bf16_गन्तव्यम्_f32;
    स्थिर constexpr ggml_bf16_t (*from_f32)(भिन्न) = f32_गन्तव्यम्_bf16;
};

प्रतिरूपम् <>
संरचना type_conversion_सारणी<int32_t> {
    स्थिर constexpr भिन्न (*गन्तव्यम्_f32)(int32_t) = i32_गन्तव्यम्_f32;
    स्थिर constexpr int32_t (*from_f32)(भिन्न) = f32_गन्तव्यम्_i32;
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

    प्रत्यागम {ir0, ir1};
}

संरचना ggml_fa_tile_config {
    स्थिर constexpr size_t Q  = GGML_FA_TILE_Q;
    स्थिर constexpr size_t KV = GGML_FA_TILE_KV;
};

#यदि_अन्तः
