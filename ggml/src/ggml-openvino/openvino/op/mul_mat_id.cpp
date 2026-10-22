#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gather_matmul.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino/ggml-openvino-extra.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/bitwise_and.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/bitwise_right_shift.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

ਨਾਮਾਕਾਸ਼ {

std::shared_ptr<ov::op::v0::Constant> const_i64(ਅਚਲ std::vector<int64_t> & ਮੁੱਲਾਂ) {
    ਵਾਪਸ ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ਮੁੱਲਾਂ.size()}, ਮੁੱਲਾਂ);
}

ov::Output<ov::Node> ਟੁਕੜਾ_axis(ਅਚਲ ov::Output<ov::Node> & ਨਿਵੇਸ਼, int64_t axis, int64_t begin, int64_t end) {
    ਵਾਪਸ std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਨਿਵੇਸ਼, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<ਪੂਰਨ> trailing_dims(ਅਚਲ ov::Output<ov::Node> & ਨਿਵੇਸ਼, ਪੂਰਨ count) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ rank = ਨਿਵੇਸ਼.get_partial_shape().rank();
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ਅਚਲ ਪੂਰਨ rank_len = static_cast<ਪੂਰਨ>(rank.get_ਲੰਮਾਈ());
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<ਪੂਰਨ> dims;
    ਲਈ (ਪੂਰਨ i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    ਵਾਪਸ dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ਅਚਲ ov::Output<ov::Node> & ਨਿਵੇਸ਼,
                                                  ਅਚਲ std::vector<ਪੂਰਨ> & dims) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ & partial_shape = ਨਿਵੇਸ਼.get_partial_shape();
    ਜੇ (partial_shape.is_static()) {
        std::vector<int64_t> ਮੁੱਲਾਂ;
        ਮੁੱਲਾਂ.reserve(dims.size());
        ਲਈ (ਅਚਲ int64_t dim : dims) {
            ਮੁੱਲਾਂ.push_back(partial_shape[dim].get_ਲੰਮਾਈ());
        }
        ਵਾਪਸ const_i64(ਮੁੱਲਾਂ);
    }

    ਸਵੈਚਾਲਿਤ shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(ਨਿਵੇਸ਼, ov::element::i64);
    ਵਾਪਸ get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ਅਚਲ ਗੰਢ_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    ਸਵੈਚਾਲਿਤ packed_shape = expert_weights.get_partial_shape().ਮੰਜ਼ਿਲ_shape();
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ਅਚਲ int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ = static_cast<int64_t>(packed_shape[2]);
    ਅਚਲ int64_t k_ਖੰਡਾਂ = static_cast<int64_t>(packed_shape[3]);
    ਅਚਲ int64_t qk = 32;
    ਅਚਲ int64_t ਖਾਨੇ_ਗਿਣਤੀ_2 = k_ਖੰਡਾਂ * qk;

    ਸਵੈਚਾਲਿਤ packed_shape_4d = const_i64({n_expert, ਪੰਕਤੀਆਂ, k_ਖੰਡਾਂ, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, ਝੂਠ);

    ਸਵੈਚਾਲਿਤ activations_shape_4d = std::make_shared<ov::op::v3::Shape_ਦਾ>(activations, ov::element::i64);
    ਸਵੈਚਾਲਿਤ ids_shape_4d = std::make_shared<ov::op::v3::Shape_ਦਾ>(ids, ov::element::i64);
    ਸਵੈਚਾਲਿਤ activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    ਸਵੈਚਾਲਿਤ ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, ਝੂਠ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, ਝੂਠ);
    ਜੇ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ਸਵੈਚਾਲਿਤ gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    ਸਥਿਰ ਅਚਲ std::vector<ਦਸ਼ਮਲਵ> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<ਦਸ਼ਮਲਵ> e8m0_lut(256);
    ਲਈ (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t ਬਿੱਟ = static_cast<uint32_t>(i) << 23;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&e8m0_lut[i], &ਬਿੱਟ, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }
    e8m0_lut[0] = std::numeric_limits<ਦਸ਼ਮਲਵ>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<ਦਸ਼ਮਲਵ>::quiet_NaN();

    ਸਵੈਚਾਲਿਤ f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    ਸਵੈਚਾਲਿਤ scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    ਸਵੈਚਾਲਿਤ selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    ਸਵੈਚਾਲਿਤ scale_byte = ਟੁਕੜਾ_axis(selected_packed_weights, 4, 0, 1);
    ਸਵੈਚਾਲਿਤ qs = ਟੁਕੜਾ_axis(selected_packed_weights, 4, 1, 17);
    ਸਵੈਚਾਲਿਤ ਨੀਵਾਂ = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
    ਸਵੈਚਾਲਿਤ ਉੱਚਾ_ਖਿਸਕਾਓ = std::make_shared<ov::op::v15::Bitwise_ਸੱਜਾ_ਖਿਸਕਾਓ>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);
    ਸਵੈਚਾਲਿਤ nibbles = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{ਨੀਵਾਂ, ਉੱਚਾ_ਖਿਸਕਾਓ}, 4);
    ਸਵੈਚਾਲਿਤ nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    ਸਵੈਚਾਲਿਤ weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    ਸਵੈਚਾਲਿਤ scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    ਸਵੈਚਾਲਿਤ scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::Auto_ਪ੍ਰਸਾਰਣ_Type::NUMPY);

    ਸਵੈਚਾਲਿਤ ids_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(ids, ov::element::i64);
    ਸਵੈਚਾਲਿਤ selected_weights_ਨਿਸ਼ਾਨਾ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ਸਦਿਸ਼{get_dimensions(ids_shape, {0, 1}), const_i64({ਪੰਕਤੀਆਂ, ਖਾਨੇ_ਗਿਣਤੀ_2})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_ਨਿਸ਼ਾਨਾ_dims, ਝੂਠ);

    ਸਵੈਚਾਲਿਤ activations_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_ਨਿਸ਼ਾਨਾ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ਸਦਿਸ਼{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(activations, acts_ਨਿਸ਼ਾਨਾ_dims, ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);

    ਸਵੈਚਾਲਿਤ activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> ਨਤੀਜਾ =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, ਝੂਠ, ਸੱਚ);

    ਸਵੈਚਾਲਿਤ batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ਸਵੈਚਾਲਿਤ ਪੰਕਤੀ_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਪੰਕਤੀਆਂ});
    ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ_ਨਿਸ਼ਾਨਾ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ਸਦਿਸ਼{batch_dim, get_dimensions(ids_shape, {0, 1}), ਪੰਕਤੀ_dim}, 0);
    ਨਤੀਜਾ = std::make_shared<ov::op::v1::Reshape>(ਨਤੀਜਾ, ਨਤੀਜਾ_ਨਿਸ਼ਾਨਾ_dims, ਝੂਠ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (ਨਤੀਜਾ.get_element_type() != output_type) {
        ਨਤੀਜਾ = std::make_shared<ov::op::v0::Convert>(ਨਤੀਜਾ, output_type);
    }
    ਵਾਪਸ ਨਤੀਜਾ;
}

}  // namespace

Output_ਸਦਿਸ਼ translate_mul_mat_id(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 3, 3);

    ਸਵੈਚਾਲਿਤ expert_weights = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ activations = process_view_ਨਿਵੇਸ਼_new(context, 1);
    ਸਵੈਚਾਲਿਤ ids = process_view_ਨਿਵੇਸ਼_new(context, 2);

    ਜੇ (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    ਜੇ (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    ਜੇ (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_ਲੰਮਾਈ() == 5) {
        ਵਾਪਸ rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    ਸਵੈਚਾਲਿਤ expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    ਜੇ (expert_weights_rank.get_ਲੰਮਾਈ() == 4) {
        ਸਵੈਚਾਲਿਤ expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, ਝੂਠ);
    }

    ਸਵੈਚਾਲਿਤ activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    ਸਵੈਚਾਲਿਤ ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, ਝੂਠ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, ਝੂਠ);

    ਜੇ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਅਚਲ ਸਵੈਚਾਲਿਤ activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    ਜੇ (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    ਸਵੈਚਾਲਿਤ activations_transpose_ਕ੍ਰਮ = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_ਕ੍ਰਮ);

    ov::Output<ov::Node> ਨਤੀਜਾ = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ_transpose_ਕ੍ਰਮ = const_i64({1, 0, 2});
    ਨਤੀਜਾ = std::make_shared<ov::op::v1::Transpose>(ਨਤੀਜਾ, ਨਤੀਜਾ_transpose_ਕ੍ਰਮ);
    ਜੇ (!context.is_stateful()) {
        ਸਵੈਚਾਲਿਤ unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਨਤੀਜਾ = std::make_shared<ov::op::v0::Unsqueeze>(ਨਤੀਜਾ, unsqueeze_axes);
    }

    ਜੇ (ਨਤੀਜਾ.get_element_type() != output_type) {
        ਨਤੀਜਾ = std::make_shared<ov::op::v0::Convert>(ਨਤੀਜਾ, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({ਨਤੀਜਾ}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
