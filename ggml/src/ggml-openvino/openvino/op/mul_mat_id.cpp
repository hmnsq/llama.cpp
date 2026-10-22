#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"
#அந்தர்பாவ "gather_matmul.hpp"
#அந்தர்பாவ "ggml-openvino/ggml-openvino-extra.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <limits>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/bitwise_and.hpp>
#அந்தர்பாவ <openvino/op/bitwise_right_shift.hpp>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

நாமாகாஶ {

std::shared_ptr<ov::op::v0::Constant> const_i64(மாறிலி std::vector<int64_t> & மதிப்புகள்) {
    திரும்பு ov::op::v0::Constant::create(ov::element::i64, ov::Shape{மதிப்புகள்.size()}, மதிப்புகள்);
}

ov::Output<ov::Node> slice_axis(மாறிலி ov::Output<ov::Node> & உள்ளீடு, int64_t axis, int64_t begin, int64_t end) {
    திரும்பு std::make_shared<ov::op::v8::Slice>(உள்ளீடு, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<முழு> trailing_dims(மாறிலி ov::Output<ov::Node> & உள்ளீடு, முழு count) {
    மாறிலி auto rank = உள்ளீடு.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    மாறிலி முழு rank_len = static_cast<முழு>(rank.get_length());
    FRONT_END_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<முழு> dims;
    க்கு (முழு i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    திரும்பு dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(மாறிலி ov::Output<ov::Node> & உள்ளீடு,
                                                  மாறிலி std::vector<முழு> & dims) {
    மாறிலி auto & partial_shape = உள்ளீடு.get_partial_shape();
    எனில் (partial_shape.is_static()) {
        std::vector<int64_t> மதிப்புகள்;
        மதிப்புகள்.reserve(dims.size());
        க்கு (மாறிலி int64_t dim : dims) {
            மதிப்புகள்.push_back(partial_shape[dim].get_length());
        }
        திரும்பு const_i64(மதிப்புகள்);
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(உள்ளீடு, ov::element::i64);
    திரும்பு get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(மாறிலி முனை_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    auto packed_shape = expert_weights.get_partial_shape().to_shape();
    FRONT_END_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    மாறிலி int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    மாறிலி int64_t rows = static_cast<int64_t>(packed_shape[2]);
    மாறிலி int64_t k_blocks = static_cast<int64_t>(packed_shape[3]);
    மாறிலி int64_t qk = 32;
    மாறிலி int64_t cols = k_blocks * qk;

    auto packed_shape_4d = const_i64({n_expert, rows, k_blocks, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, பொய்);

    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, பொய்);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, பொய்);
    எனில் (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    நிலையான மாறிலி std::vector<தசம> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<தசம> e8m0_lut(256);
    க்கு (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i) << 23;
        memcpy(&e8m0_lut[i], &bits, அளவு(தசம));
    }
    e8m0_lut[0] = std::numeric_limits<தசம>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<தசம>::quiet_NaN();

    auto f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    auto scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    auto selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    auto scale_byte = slice_axis(selected_packed_weights, 4, 0, 1);
    auto qs = slice_axis(selected_packed_weights, 4, 1, 17);
    auto கீழ் = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::AutoBroadcastType::NUMPY);
    auto உயர்வு_shift = std::make_shared<ov::op::v15::Bitwise_வலது_Shift>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::AutoBroadcastType::NUMPY);
    auto nibbles = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{கீழ், உயர்வு_shift}, 4);
    auto nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    auto weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    auto scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    auto scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::AutoBroadcastType::NUMPY);

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), const_i64({rows, cols})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, பொய்);

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
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, பொய், உண்மை);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {rows});
    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim}, 0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, பொய்);

    மாறிலி auto output_type = context.get_output_type();
    எனில் (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    திரும்பு result;
}

}  // namespace

OutputVector translate_mul_mat_id(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 3, 3);

    auto expert_weights = process_view_உள்ளீடு_new(context, 0);
    auto activations = process_view_உள்ளீடு_new(context, 1);
    auto ids = process_view_உள்ளீடு_new(context, 2);

    எனில் (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    எனில் (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    எனில் (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_length() == 5) {
        திரும்பு rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    எனில் (expert_weights_rank.get_length() == 4) {
        auto expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, பொய்);
    }

    auto activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    auto ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, பொய்);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, பொய்);

    எனில் (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    மாறிலி auto output_type = context.get_output_type();
    மாறிலி auto activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    எனில் (activations.get_element_type() != activations_type) {
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
    எனில் (!context.is_stateful()) {
        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        result = std::make_shared<ov::op::v0::Unsqueeze>(result, unsqueeze_axes);
    }

    எனில் (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }

    திரும்பு rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
