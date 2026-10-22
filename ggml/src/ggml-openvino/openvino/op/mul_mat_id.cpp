#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"
#समावेश "gather_matmul.hpp"
#समावेश "ggml-openvino/ggml-openvino-extra.h"

#समावेश <cstdint>
#समावेश <cstring>
#समावेश <limits>
#समावेश <memory>
#समावेश <openvino/op/bitwise_and.hpp>
#समावेश <openvino/op/bitwise_right_shift.hpp>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/matmul.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <openvino/op/unsqueeze.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

std::shared_ptr<ov::op::v0::Constant> const_i64(स्थिर std::vector<int64_t> & मूल्यावली) {
    वापस ov::op::v0::Constant::create(ov::element::i64, ov::Shape{मूल्यावली.size()}, मूल्यावली);
}

ov::Output<ov::Node> slice_axis(स्थिर ov::Output<ov::Node> & निवेश, int64_t axis, int64_t begin, int64_t end) {
    वापस std::make_shared<ov::op::v8::Slice>(निवेश, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<पूर्णांक> trailing_dims(स्थिर ov::Output<ov::Node> & निवेश, पूर्णांक count) {
    स्थिर स्वतः rank = निवेश.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    स्थिर पूर्णांक rank_len = static_cast<पूर्णांक>(rank.get_लंबाई());
    FRONT_END_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<पूर्णांक> dims;
    क्रम (पूर्णांक i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    वापस dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(स्थिर ov::Output<ov::Node> & निवेश,
                                                  स्थिर std::vector<पूर्णांक> & dims) {
    स्थिर स्वतः & partial_shape = निवेश.get_partial_shape();
    अगर (partial_shape.is_static()) {
        std::vector<int64_t> मूल्यावली;
        मूल्यावली.reserve(dims.size());
        क्रम (स्थिर int64_t dim : dims) {
            मूल्यावली.push_back(partial_shape[dim].get_लंबाई());
        }
        वापस const_i64(मूल्यावली);
    }

    स्वतः shape = std::make_shared<ov::op::v3::ShapeOf>(निवेश, ov::element::i64);
    वापस get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(स्थिर गाँठ_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    स्वतः packed_shape = expert_weights.get_partial_shape().to_shape();
    FRONT_END_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    स्थिर int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    स्थिर int64_t rows = static_cast<int64_t>(packed_shape[2]);
    स्थिर int64_t k_blocks = static_cast<int64_t>(packed_shape[3]);
    स्थिर int64_t qk = 32;
    स्थिर int64_t cols = k_blocks * qk;

    स्वतः packed_shape_4d = const_i64({n_expert, rows, k_blocks, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, false);

    स्वतः activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    स्वतः ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    स्वतः activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    स्वतः ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);
    अगर (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    स्वतः gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    अटल स्थिर std::vector<भग्न> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<भग्न> e8m0_lut(256);
    क्रम (माप_प्रकार i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i) << 23;
        स_नकल(&e8m0_lut[i], &bits, माप(भग्न));
    }
    e8m0_lut[0] = std::numeric_limits<भग्न>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<भग्न>::quiet_NaN();

    स्वतः f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    स्वतः scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    स्वतः selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    स्वतः scale_byte = slice_axis(selected_packed_weights, 4, 0, 1);
    स्वतः qs = slice_axis(selected_packed_weights, 4, 1, 17);
    स्वतः निचला = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::AutoBroadcastType::NUMPY);
    स्वतः ऊपरी_shift = std::make_shared<ov::op::v15::Bitwise_दायाँ_Shift>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::AutoBroadcastType::NUMPY);
    स्वतः nibbles = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{निचला, ऊपरी_shift}, 4);
    स्वतः nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    स्वतः weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    स्वतः scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    स्वतः scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::AutoBroadcastType::NUMPY);

    स्वतः ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    स्वतः selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), const_i64({rows, cols})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, false);

    स्वतः activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims, ov::op::BroadcastType::BIDIRECTIONAL);

    स्वतः activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> परिणाम =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);

    स्वतः batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {rows});
    स्वतः परिणाम_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim}, 0);
    परिणाम = std::make_shared<ov::op::v1::Reshape>(परिणाम, परिणाम_target_dims, false);

    स्थिर स्वतः output_type = context.get_output_type();
    अगर (परिणाम.get_element_type() != output_type) {
        परिणाम = std::make_shared<ov::op::v0::Convert>(परिणाम, output_type);
    }
    वापस परिणाम;
}

}  // namespace

OutputVector translate_mul_mat_id(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 3, 3);

    स्वतः expert_weights = process_view_निवेश_new(context, 0);
    स्वतः activations = process_view_निवेश_new(context, 1);
    स्वतः ids = process_view_निवेश_new(context, 2);

    अगर (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    अगर (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    अगर (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_लंबाई() == 5) {
        वापस rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    स्वतः expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    अगर (expert_weights_rank.get_लंबाई() == 4) {
        स्वतः expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, false);
    }

    स्वतः activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    स्वतः ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);

    अगर (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    स्थिर स्वतः output_type = context.get_output_type();
    स्थिर स्वतः activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    अगर (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    स्वतः activations_transpose_order = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_order);

    ov::Output<ov::Node> परिणाम = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    स्वतः परिणाम_transpose_order = const_i64({1, 0, 2});
    परिणाम = std::make_shared<ov::op::v1::Transpose>(परिणाम, परिणाम_transpose_order);
    अगर (!context.is_stateful()) {
        स्वतः unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        परिणाम = std::make_shared<ov::op::v0::Unsqueeze>(परिणाम, unsqueeze_axes);
    }

    अगर (परिणाम.get_element_type() != output_type) {
        परिणाम = std::make_shared<ov::op::v0::Convert>(परिणाम, output_type);
    }

    वापस rename_outputs_with_suffix({परिणाम}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
