#pragma once

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "traits.h"
#અન્તર્ભાવ "ggml-cpu-impl.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#અન્તર્ભાવ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
સ્થિર અંતઃરેખ ggml_fp16_t f32_to_f16(દશાંશ x) {
    પાછા GGML_CPU_FP32_TO_FP16(x);
}

સ્થિર અંતઃરેખ દશાંશ f16_to_f32(ggml_fp16_t x) {
    પાછા GGML_CPU_FP16_TO_FP32(x);
}

સ્થિર અંતઃરેખ ggml_bf16_t f32_to_bf16(દશાંશ x) {
    પાછા GGML_FP32_TO_BF16(x);
}

સ્થિર અંતઃરેખ દશાંશ bf16_to_f32(ggml_bf16_t x) {
    પાછા GGML_BF16_TO_FP32(x);
}

સ્થિર અંતઃરેખ દશાંશ i32_to_f32(int32_t x) {
    પાછા x;
}

સ્થિર અંતઃરેખ int32_t f32_to_i32(દશાંશ x) {
    પાછા x;
}

સ્થિર અંતઃરેખ દશાંશ f32_to_f32(દશાંશ x) {
    પાછા x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
સંરચના type_conversion_table;

template <>
સંરચના type_conversion_table<ggml_fp16_t> {
    સ્થિર constexpr દશાંશ (*to_f32)(ggml_fp16_t) = f16_to_f32;
    સ્થિર constexpr ggml_fp16_t (*from_f32)(દશાંશ) = f32_to_f16;
};

template <>
સંરચના type_conversion_table<દશાંશ> {
    સ્થિર constexpr દશાંશ (*to_f32)(દશાંશ) = f32_to_f32;
    સ્થિર constexpr દશાંશ (*from_f32)(દશાંશ) = f32_to_f32;
};

template <>
સંરચના type_conversion_table<ggml_bf16_t> {
    સ્થિર constexpr દશાંશ (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    સ્થિર constexpr ggml_bf16_t (*from_f32)(દશાંશ) = f32_to_bf16;
};

template <>
સંરચના type_conversion_table<int32_t> {
    સ્થિર constexpr દશાંશ (*to_f32)(int32_t) = i32_to_f32;
    સ્થિર constexpr int32_t (*from_f32)(દશાંશ) = f32_to_i32;
};

સ્થિર std::pair<int64_t, int64_t> get_thread_range(અચલ સંરચના ggml_compute_params * params, અચલ સંરચના ggml_tensor * src0) {
    અચલ int64_t ith = params->ith;
    અચલ int64_t nth = params->nth;

    અચલ int64_t nr  = ggml_nrows(src0);

    // rows per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    પાછા {ir0, ir1};
}

સંરચના ggml_fa_tile_config {
    સ્થિર constexpr size_t Q  = GGML_FA_TILE_Q;
    સ્થિર constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
