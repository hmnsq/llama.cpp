#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"
#অন্তর্ভাৱ "gather_matmul.hpp"
#অন্তর্ভাৱ "ggml-openvino/ggml-openvino-extra.h"

#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <limits>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/bitwise_and.hpp>
#অন্তর্ভাৱ <openvino/op/bitwise_right_shift.hpp>
#অন্তর্ভাৱ <openvino/op/broadcast.hpp>
#অন্তর্ভাৱ <openvino/op/concat.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>
#অন্তর্ভাৱ <openvino/op/gather.hpp>
#অন্তর্ভাৱ <openvino/op/matmul.hpp>
#অন্তর্ভাৱ <openvino/op/multiply.hpp>
#অন্তর্ভাৱ <openvino/op/reshape.hpp>
#অন্তর্ভাৱ <openvino/op/shape_of.hpp>
#অন্তর্ভাৱ <openvino/op/slice.hpp>
#অন্তর্ভাৱ <openvino/op/transpose.hpp>
#অন্তর্ভাৱ <openvino/op/unsqueeze.hpp>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

নামাকাশ {

std::shared_ptr<ov::op::v0::Constant> const_i64(ধ্রুৱ std::vector<int64_t> & মানসমূহ) {
    ঘূৰাই ov::op::v0::Constant::create(ov::element::i64, ov::Shape{মানসমূহ.size()}, মানসমূহ);
}

ov::Output<ov::Node> slice_axis(ধ্রুৱ ov::Output<ov::Node> & প্ৰবেশ, int64_t axis, int64_t begin, int64_t end) {
    ঘূৰাই std::make_shared<ov::op::v8::Slice>(প্ৰবেশ, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<পূর্ণাঙ্ক> trailing_dims(ধ্রুৱ ov::Output<ov::Node> & প্ৰবেশ, পূর্ণাঙ্ক count) {
    ধ্রুৱ auto rank = প্ৰবেশ.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ধ্রুৱ পূর্ণাঙ্ক rank_len = static_cast<পূর্ণাঙ্ক>(rank.get_length());
    FRONT_END_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<পূর্ণাঙ্ক> dims;
    বাবে (পূর্ণাঙ্ক i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    ঘূৰাই dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ধ্রুৱ ov::Output<ov::Node> & প্ৰবেশ,
                                                  ধ্রুৱ std::vector<পূর্ণাঙ্ক> & dims) {
    ধ্রুৱ auto & partial_shape = প্ৰবেশ.get_partial_shape();
    যদি (partial_shape.is_static()) {
        std::vector<int64_t> মানসমূহ;
        মানসমূহ.reserve(dims.size());
        বাবে (ধ্রুৱ int64_t dim : dims) {
            মানসমূহ.push_back(partial_shape[dim].get_length());
        }
        ঘূৰাই const_i64(মানসমূহ);
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(প্ৰবেশ, ov::element::i64);
    ঘূৰাই get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ধ্রুৱ গ্ৰন্থি_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    auto packed_shape = expert_weights.get_partial_shape().to_shape();
    FRONT_END_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ধ্রুৱ int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ধ্রুৱ int64_t rows = static_cast<int64_t>(packed_shape[2]);
    ধ্রুৱ int64_t k_blocks = static_cast<int64_t>(packed_shape[3]);
    ধ্রুৱ int64_t qk = 32;
    ধ্রুৱ int64_t cols = k_blocks * qk;

    auto packed_shape_4d = const_i64({n_expert, rows, k_blocks, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, মিছা);

    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, মিছা);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, মিছা);
    যদি (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    স্থির ধ্রুৱ std::vector<ভিন্ন> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<ভিন্ন> e8m0_lut(256);
    বাবে (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i) << 23;
        memcpy(&e8m0_lut[i], &bits, পরিমাণ(ভিন্ন));
    }
    e8m0_lut[0] = std::numeric_limits<ভিন্ন>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<ভিন্ন>::quiet_NaN();

    auto f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    auto scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    auto selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    auto scale_byte = slice_axis(selected_packed_weights, 4, 0, 1);
    auto qs = slice_axis(selected_packed_weights, 4, 1, 17);
    auto নিচেৰ = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::AutoBroadcastType::NUMPY);
    auto উপৰেৰ_shift = std::make_shared<ov::op::v15::Bitwise_ডান_Shift>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::AutoBroadcastType::NUMPY);
    auto nibbles = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{নিচেৰ, উপৰেৰ_shift}, 4);
    auto nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    auto weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    auto scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    auto scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::AutoBroadcastType::NUMPY);

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), const_i64({rows, cols})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, মিছা);

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
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, মিছা, সঁচা);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {rows});
    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim}, 0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, মিছা);

    ধ্রুৱ auto output_type = context.get_output_type();
    যদি (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    ঘূৰাই result;
}

}  // namespace

OutputVector translate_mul_mat_id(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 3, 3);

    auto expert_weights = process_view_প্ৰবেশ_new(context, 0);
    auto activations = process_view_প্ৰবেশ_new(context, 1);
    auto ids = process_view_প্ৰবেশ_new(context, 2);

    যদি (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    যদি (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    যদি (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_length() == 5) {
        ঘূৰাই rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    যদি (expert_weights_rank.get_length() == 4) {
        auto expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, মিছা);
    }

    auto activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    auto ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, মিছা);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, মিছা);

    যদি (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ধ্রুৱ auto output_type = context.get_output_type();
    ধ্রুৱ auto activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    যদি (activations.get_element_type() != activations_type) {
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
    যদি (!context.is_stateful()) {
        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        result = std::make_shared<ov::op::v0::Unsqueeze>(result, unsqueeze_axes);
    }

    যদি (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }

    ঘূৰাই rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
