#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"
#ಅನ್ತರ್ಭಾವ "gather_matmul.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/bitwise_and.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/bitwise_right_shift.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

ನಾಮಾಕಾಶ {

std::shared_ptr<ov::op::v0::Constant> const_i64(ಸ್ಥಿರಾಂಕ std::vector<int64_t> & ಮೌಲ್ಯಗಳು) {
    ಹಿಂತಿರುಗಿ ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ಮೌಲ್ಯಗಳು.size()}, ಮೌಲ್ಯಗಳು);
}

ov::Output<ov::Node> ತುಂಡು_axis(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & ಒಳಹರಿವು, int64_t axis, int64_t begin, int64_t end) {
    ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v8::ತುಂಡು>(ಒಳಹರಿವು, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<ಪೂರ್ಣಾಂಕ> trailing_dims(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & ಒಳಹರಿವು, ಪೂರ್ಣಾಂಕ count) {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rank = ಒಳಹರಿವು.get_partial_shape().rank();
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rank_len = static_cast<ಪೂರ್ಣಾಂಕ>(rank.get_ಉದ್ದಳತೆ());
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<ಪೂರ್ಣಾಂಕ> dims;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    ಹಿಂತಿರುಗಿ dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & ಒಳಹರಿವು,
                                                  ಸ್ಥಿರಾಂಕ std::vector<ಪೂರ್ಣಾಂಕ> & dims) {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & partial_shape = ಒಳಹರಿವು.get_partial_shape();
    ಆದರೆ (partial_shape.is_static()) {
        std::vector<int64_t> ಮೌಲ್ಯಗಳು;
        ಮೌಲ್ಯಗಳು.reserve(dims.size());
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ int64_t dim : dims) {
            ಮೌಲ್ಯಗಳು.push_back(partial_shape[dim].get_ಉದ್ದಳತೆ());
        }
        ಹಿಂತಿರುಗಿ const_i64(ಮೌಲ್ಯಗಳು);
    }

    ಸ್ವಯಂಚಾಲಿತ shape = std::make_shared<ov::op::v3::Shape_ದ>(ಒಳಹರಿವು, ov::element::i64);
    ಹಿಂತಿರುಗಿ get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    ಸ್ವಯಂಚಾಲಿತ packed_shape = expert_weights.get_partial_shape().ಗಮ್ಯ_shape();
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ಸ್ಥಿರಾಂಕ int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು = static_cast<int64_t>(packed_shape[2]);
    ಸ್ಥಿರಾಂಕ int64_t k_ಖಂಡಗಳು = static_cast<int64_t>(packed_shape[3]);
    ಸ್ಥಿರಾಂಕ int64_t qk = 32;
    ಸ್ಥಿರಾಂಕ int64_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 = k_ಖಂಡಗಳು * qk;

    ಸ್ವಯಂಚಾಲಿತ packed_shape_4d = const_i64({n_expert, ಅಡ್ಡಸಾಲುಗಳು, k_ಖಂಡಗಳು, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ activations_shape_4d = std::make_shared<ov::op::v3::Shape_ದ>(activations, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ ids_shape_4d = std::make_shared<ov::op::v3::Shape_ದ>(ids, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    ಸ್ವಯಂಚಾಲಿತ ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, ಸುಳ್ಳು);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, ಸುಳ್ಳು);
    ಆದರೆ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ಸ್ವಯಂಚಾಲಿತ gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ದಶಮಾಂಶ> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<ದಶಮಾಂಶ> e8m0_lut(256);
    ಗಾಗಿ (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t ಬಿಟ್ಟುಗಳು = static_cast<uint32_t>(i) << 23;
        ಸ್ಮರಣೆ_ನಕಲು(&e8m0_lut[i], &ಬಿಟ್ಟುಗಳು, ಗಾತ್ರ(ದಶಮಾಂಶ));
    }
    e8m0_lut[0] = std::numeric_limits<ದಶಮಾಂಶ>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<ದಶಮಾಂಶ>::quiet_NaN();

    ಸ್ವಯಂಚಾಲಿತ f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    ಸ್ವಯಂಚಾಲಿತ scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    ಸ್ವಯಂಚಾಲಿತ selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    ಸ್ವಯಂಚಾಲಿತ scale_byte = ತುಂಡು_axis(selected_packed_weights, 4, 0, 1);
    ಸ್ವಯಂಚಾಲಿತ qs = ತುಂಡು_axis(selected_packed_weights, 4, 1, 17);
    ಸ್ವಯಂಚಾಲಿತ ಕೆಳಗಿನ = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
    ಸ್ವಯಂಚಾಲಿತ ಮೇಲಿನ_ಸರಿಸು = std::make_shared<ov::op::v15::Bitwise_ಬಲ_ಸರಿಸು>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);
    ಸ್ವಯಂಚಾಲಿತ nibbles = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{ಕೆಳಗಿನ, ಮೇಲಿನ_ಸರಿಸು}, 4);
    ಸ್ವಯಂಚಾಲಿತ nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    ಸ್ವಯಂಚಾಲಿತ weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    ಸ್ವಯಂಚಾಲಿತ scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    ಸ್ವಯಂಚಾಲಿತ scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::Auto_ಪ್ರಸಾರ_Type::NUMPY);

    ಸ್ವಯಂಚಾಲಿತ ids_shape = std::make_shared<ov::op::v3::Shape_ದ>(ids, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ selected_weights_ಗುರಿ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ಸದಿಶ{get_dimensions(ids_shape, {0, 1}), const_i64({ಅಡ್ಡಸಾಲುಗಳು, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_ಗುರಿ_dims, ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ activations_shape = std::make_shared<ov::op::v3::Shape_ದ>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_ಗುರಿ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ಸದಿಶ{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(activations, acts_ಗುರಿ_dims, ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);

    ಸ್ವಯಂಚಾಲಿತ activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> ಫಲಿತಾಂಶ =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, ಸುಳ್ಳು, ನಿಜ);

    ಸ್ವಯಂಚಾಲಿತ batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ಸ್ವಯಂಚಾಲಿತ ಅಡ್ಡಸಾಲು_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {ಅಡ್ಡಸಾಲುಗಳು});
    ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ_ಗುರಿ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ಸದಿಶ{batch_dim, get_dimensions(ids_shape, {0, 1}), ಅಡ್ಡಸಾಲು_dim}, 0);
    ಫಲಿತಾಂಶ = std::make_shared<ov::op::v1::Reshape>(ಫಲಿತಾಂಶ, ಫಲಿತಾಂಶ_ಗುರಿ_dims, ಸುಳ್ಳು);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (ಫಲಿತಾಂಶ.get_element_type() != output_type) {
        ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Convert>(ಫಲಿತಾಂಶ, output_type);
    }
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

}  // namespace

Output_ಸದಿಶ translate_mul_mat_id(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 3, 3);

    ಸ್ವಯಂಚಾಲಿತ expert_weights = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ activations = process_view_ಒಳಹರಿವು_new(context, 1);
    ಸ್ವಯಂಚಾಲಿತ ids = process_view_ಒಳಹರಿವು_new(context, 2);

    ಆದರೆ (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    ಆದರೆ (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    ಆದರೆ (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_ಉದ್ದಳತೆ() == 5) {
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    ಸ್ವಯಂಚಾಲಿತ expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    ಆದರೆ (expert_weights_rank.get_ಉದ್ದಳತೆ() == 4) {
        ಸ್ವಯಂಚಾಲಿತ expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, ಸುಳ್ಳು);
    }

    ಸ್ವಯಂಚಾಲಿತ activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    ಸ್ವಯಂಚಾಲಿತ ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, ಸುಳ್ಳು);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, ಸುಳ್ಳು);

    ಆದರೆ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    ಆದರೆ (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    ಸ್ವಯಂಚಾಲಿತ activations_transpose_ಕ್ರಮ = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_ಕ್ರಮ);

    ov::Output<ov::Node> ಫಲಿತಾಂಶ = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ_transpose_ಕ್ರಮ = const_i64({1, 0, 2});
    ಫಲಿತಾಂಶ = std::make_shared<ov::op::v1::Transpose>(ಫಲಿತಾಂಶ, ಫಲಿತಾಂಶ_transpose_ಕ್ರಮ);
    ಆದರೆ (!context.is_stateful()) {
        ಸ್ವಯಂಚಾಲಿತ unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Unsqueeze>(ಫಲಿತಾಂಶ, unsqueeze_axes);
    }

    ಆದರೆ (ಫಲಿತಾಂಶ.get_element_type() != output_type) {
        ಫಲಿತಾಂಶ = std::make_shared<ov::op::v0::Convert>(ಫಲಿತಾಂಶ, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({ಫಲಿತಾಂಶ}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
