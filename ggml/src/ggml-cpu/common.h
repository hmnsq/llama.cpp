#आशय once

#समावेश "ggml.h"
#समावेश "traits.h"
#समावेश "ggml-cpu-impl.h"
#समावेश "ggml-impl.h"
#समावेश "simd-mappings.h"

#घोषणा GGML_FA_TILE_Q  64
#घोषणा GGML_FA_TILE_KV 64

#अगर_घोषित __cplusplus

#समावेश <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
अटल अंतरभूत ggml_fp16_t f32_to_f16(भग्न x) {
    वापस GGML_CPU_FP32_TO_FP16(x);
}

अटल अंतरभूत भग्न f16_to_f32(ggml_fp16_t x) {
    वापस GGML_CPU_FP16_TO_FP32(x);
}

अटल अंतरभूत ggml_bf16_t f32_to_bf16(भग्न x) {
    वापस GGML_FP32_TO_BF16(x);
}

अटल अंतरभूत भग्न bf16_to_f32(ggml_bf16_t x) {
    वापस GGML_BF16_TO_FP32(x);
}

अटल अंतरभूत भग्न i32_to_f32(int32_t x) {
    वापस x;
}

अटल अंतरभूत int32_t f32_to_i32(भग्न x) {
    वापस x;
}

अटल अंतरभूत भग्न f32_to_f32(भग्न x) {
    वापस x;
}

// TODO - merge this into the traits table, after using row-based conversions
ढाँचा <class T>
काष्ठा type_conversion_table;

ढाँचा <>
काष्ठा type_conversion_table<ggml_fp16_t> {
    अटल constexpr भग्न (*to_f32)(ggml_fp16_t) = f16_to_f32;
    अटल constexpr ggml_fp16_t (*from_f32)(भग्न) = f32_to_f16;
};

ढाँचा <>
काष्ठा type_conversion_table<भग्न> {
    अटल constexpr भग्न (*to_f32)(भग्न) = f32_to_f32;
    अटल constexpr भग्न (*from_f32)(भग्न) = f32_to_f32;
};

ढाँचा <>
काष्ठा type_conversion_table<ggml_bf16_t> {
    अटल constexpr भग्न (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    अटल constexpr ggml_bf16_t (*from_f32)(भग्न) = f32_to_bf16;
};

ढाँचा <>
काष्ठा type_conversion_table<int32_t> {
    अटल constexpr भग्न (*to_f32)(int32_t) = i32_to_f32;
    अटल constexpr int32_t (*from_f32)(भग्न) = f32_to_i32;
};

अटल std::pair<int64_t, int64_t> get_thread_range(स्थिर काष्ठा ggml_compute_params * params, स्थिर काष्ठा ggml_tensor * src0) {
    स्थिर int64_t ith = params->ith;
    स्थिर int64_t nth = params->nth;

    स्थिर int64_t nr  = ggml_nrows(src0);

    // rows per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    वापस {ir0, ir1};
}

काष्ठा ggml_fa_tile_config {
    अटल constexpr माप_प्रकार Q  = GGML_FA_TILE_Q;
    अटल constexpr माप_प्रकार KV = GGML_FA_TILE_KV;
};

#पूर्ण_अगर
