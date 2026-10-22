#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "gather_matmul.hpp"
#अन्तर्भाव "ggml-openvino/ggml-openvino-extra.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstring>
#अन्तर्भाव <limits>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/bitwise_and.hpp>
#अन्तर्भाव <openvino/op/bitwise_right_shift.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

नामाकाश {

std::shared_ptr<ov::op::v0::Constant> const_i64(ध्रुव std::vector<int64_t> & मानानि) {
    प्रत्यागम ov::op::v0::Constant::create(ov::element::i64, ov::Shape{मानानि.size()}, मानानि);
}

ov::Output<ov::Node> slice_axis(ध्रुव ov::Output<ov::Node> & निवेशः, int64_t axis, int64_t begin, int64_t end) {
    प्रत्यागम std::make_shared<ov::op::v8::Slice>(निवेशः, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<पूर्णाङ्क> trailing_dims(ध्रुव ov::Output<ov::Node> & निवेशः, पूर्णाङ्क count) {
    ध्रुव auto rank = निवेशः.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ध्रुव पूर्णाङ्क rank_len = static_cast<पूर्णाङ्क>(rank.get_length());
    FRONT_END_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<पूर्णाङ्क> dims;
    प्रति (पूर्णाङ्क i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    प्रत्यागम dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ध्रुव ov::Output<ov::Node> & निवेशः,
                                                  ध्रुव std::vector<पूर्णाङ्क> & dims) {
    ध्रुव auto & partial_shape = निवेशः.get_partial_shape();
    जइ (partial_shape.is_static()) {
        std::vector<int64_t> मानानि;
        मानानि.reserve(dims.size());
        प्रति (ध्रुव int64_t dim : dims) {
            मानानि.push_back(partial_shape[dim].get_length());
        }
        प्रत्यागम const_i64(मानानि);
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(निवेशः, ov::element::i64);
    प्रत्यागम get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ध्रुव ग्रन्थिः_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    auto packed_shape = expert_weights.get_partial_shape().to_shape();
    FRONT_END_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ध्रुव int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ध्रुव int64_t rows = static_cast<int64_t>(packed_shape[2]);
    ध्रुव int64_t k_blocks = static_cast<int64_t>(packed_shape[3]);
    ध्रुव int64_t qk = 32;
    ध्रुव int64_t cols = k_blocks * qk;

    auto packed_shape_4d = const_i64({n_expert, rows, k_blocks, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, मिच्छ);

    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, मिच्छ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, मिच्छ);
    जइ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    स्थिर ध्रुव std::vector<भिन्न> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<भिन्न> e8m0_lut(256);
    प्रति (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i) << 23;
        memcpy(&e8m0_lut[i], &bits, परिमाण(भिन्न));
    }
    e8m0_lut[0] = std::numeric_limits<भिन्न>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<भिन्न>::quiet_NaN();

    auto f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    auto scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    auto selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    auto scale_byte = slice_axis(selected_packed_weights, 4, 0, 1);
    auto qs = slice_axis(selected_packed_weights, 4, 1, 17);
    auto नीचः = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::AutoBroadcastType::NUMPY);
    auto उच्चः_shift = std::make_shared<ov::op::v15::Bitwise_दक्षिणः_Shift>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::AutoBroadcastType::NUMPY);
    auto nibbles = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{नीचः, उच्चः_shift}, 4);
    auto nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    auto weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    auto scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    auto scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::AutoBroadcastType::NUMPY);

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), const_i64({rows, cols})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, मिच्छ);

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims, ov::op::BroadcastType::BIDIRECTIONAL);

    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, मिच्छ, सच्च);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {rows});
    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim}, 0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, मिच्छ);

    ध्रुव auto output_type = context.get_output_type();
    जइ (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    प्रत्यागम result;
}

}  // namespace

OutputVector translate_mul_mat_id(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 3, 3);

    auto expert_weights = process_view_निवेशः_new(context, 0);
    auto activations = process_view_निवेशः_new(context, 1);
    auto ids = process_view_निवेशः_new(context, 2);

    जइ (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    जइ (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    जइ (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_length() == 5) {
        प्रत्यागम rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
                                          context.get_name());
    }

    // General (non-packed) path: dense F32/F16/BF16 weights, or the f16 dequantization chain for
    // quantized MoE experts (see extract_quantized_weights / make_int4_weights / make_int8_weights in
    // ggml-quants.cpp). Routed through ov::op::internal::GatherMatmul instead of a naive
    // Gather+Broadcast+MatMul, so the selected expert's full weight matrix is never materialized per
    // token. The CPU plugin's ConvertGatherMatmulToGatherMatmulCompressed pass (run during
    // compile_model) fuses the dequantization chain feeding GatherMatmul's B input into a
    // GatherMatmulCompressed node automatically, as long as MarkDequantization has marked the chain --
    // see translate_session.cpp's apply_transformations for the MarkDequantization registration.
    //
    // OpenVINO sees GGML tensors in reversed dimension order:
    //   activations: [1, n_tokens, n_used_or_1, k]
    //   ids: [1, 1, n_tokens, n_used]
    // expert_weights is either [1, n_expert, m, k] (4D, e.g. non-quantized weights without a
    // pre-built extra) or already [n_expert, m, k] (3D, weights routed through
    // process_weight_tensor) -- GatherMatmul's B input expects the latter.
    auto expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    जइ (expert_weights_rank.get_length() == 4) {
        auto expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, मिच्छ);
    }

    auto activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    auto ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, मिच्छ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, मिच्छ);

    जइ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ध्रुव auto output_type = context.get_output_type();
    ध्रुव auto activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    जइ (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    auto activations_transpose_order = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_order);

    ov::Output<ov::Node> result = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    auto result_transpose_order = const_i64({1, 0, 2});
    result = std::make_shared<ov::op::v1::Transpose>(result, result_transpose_order);
    जइ (!context.is_stateful()) {
        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        result = std::make_shared<ov::op::v0::Unsqueeze>(result, unsqueeze_axes);
    }

    जइ (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
