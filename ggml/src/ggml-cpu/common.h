#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "traits.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#ଅନ୍ତର୍ଭାଵ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_fp16_t f32_to_f16(ଦଶମିକ x) {
    ଫେରନ୍ତୁ GGML_CPU_FP32_TO_FP16(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ f16_to_f32(ggml_fp16_t x) {
    ଫେରନ୍ତୁ GGML_CPU_FP16_TO_FP32(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_bf16_t f32_to_bf16(ଦଶମିକ x) {
    ଫେରନ୍ତୁ GGML_FP32_TO_BF16(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ bf16_to_f32(ggml_bf16_t x) {
    ଫେରନ୍ତୁ GGML_BF16_TO_FP32(x);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ i32_to_f32(int32_t x) {
    ଫେରନ୍ତୁ x;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int32_t f32_to_i32(ଦଶମିକ x) {
    ଫେରନ୍ତୁ x;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ f32_to_f32(ଦଶମିକ x) {
    ଫେରନ୍ତୁ x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
ସଂରଚନା type_conversion_table;

template <>
ସଂରଚନା type_conversion_table<ggml_fp16_t> {
    ସ୍ଥିର constexpr ଦଶମିକ (*to_f32)(ggml_fp16_t) = f16_to_f32;
    ସ୍ଥିର constexpr ggml_fp16_t (*from_f32)(ଦଶମିକ) = f32_to_f16;
};

template <>
ସଂରଚନା type_conversion_table<ଦଶମିକ> {
    ସ୍ଥିର constexpr ଦଶମିକ (*to_f32)(ଦଶମିକ) = f32_to_f32;
    ସ୍ଥିର constexpr ଦଶମିକ (*from_f32)(ଦଶମିକ) = f32_to_f32;
};

template <>
ସଂରଚନା type_conversion_table<ggml_bf16_t> {
    ସ୍ଥିର constexpr ଦଶମିକ (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    ସ୍ଥିର constexpr ggml_bf16_t (*from_f32)(ଦଶମିକ) = f32_to_bf16;
};

template <>
ସଂରଚନା type_conversion_table<int32_t> {
    ସ୍ଥିର constexpr ଦଶମିକ (*to_f32)(int32_t) = i32_to_f32;
    ସ୍ଥିର constexpr int32_t (*from_f32)(ଦଶମିକ) = f32_to_i32;
};

ସ୍ଥିର std::pair<int64_t, int64_t> get_thread_range(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ଅଚଳ ସଂରଚନା ggml_tensor * src0) {
    ଅଚଳ int64_t ith = params->ith;
    ଅଚଳ int64_t nth = params->nth;

    ଅଚଳ int64_t nr  = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ଫେରନ୍ତୁ {ir0, ir1};
}

ସଂରଚନା ggml_fa_tile_config {
    ସ୍ଥିର constexpr size_t Q  = GGML_FA_TILE_Q;
    ସ୍ଥିର constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
