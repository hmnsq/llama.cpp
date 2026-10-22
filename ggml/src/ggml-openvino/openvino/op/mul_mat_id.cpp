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

ov::Output<ov::Node> खण्डिका_axis(ध्रुव ov::Output<ov::Node> & निवेशः, int64_t axis, int64_t begin, int64_t end) {
    प्रत्यागम std::make_shared<ov::op::v8::खण्डिका>(निवेशः, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<पूर्णाङ्क> trailing_dims(ध्रुव ov::Output<ov::Node> & निवेशः, पूर्णाङ्क count) {
    ध्रुव स्वचालित rank = निवेशः.get_partial_shape().rank();
    FRONT_अन्तः_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ध्रुव पूर्णाङ्क rank_len = static_cast<पूर्णाङ्क>(rank.get_दीर्घता());
    FRONT_अन्तः_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<पूर्णाङ्क> dims;
    प्रति (पूर्णाङ्क i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    प्रत्यागम dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ध्रुव ov::Output<ov::Node> & निवेशः,
                                                  ध्रुव std::vector<पूर्णाङ्क> & dims) {
    ध्रुव स्वचालित & partial_shape = निवेशः.get_partial_shape();
    यदि (partial_shape.is_static()) {
        std::vector<int64_t> मानानि;
        मानानि.reserve(dims.size());
        प्रति (ध्रुव int64_t dim : dims) {
            मानानि.push_back(partial_shape[dim].get_दीर्घता());
        }
        प्रत्यागम const_i64(मानानि);
    }

    स्वचालित shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(निवेशः, ov::element::i64);
    प्रत्यागम get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ध्रुव ग्रन्थिः_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    स्वचालित packed_shape = expert_weights.get_partial_shape().गन्तव्यम्_shape();
    FRONT_अन्तः_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ध्रुव int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ध्रुव int64_t पङ्क्तिकाः = static_cast<int64_t>(packed_shape[2]);
    ध्रुव int64_t k_खण्डानि = static_cast<int64_t>(packed_shape[3]);
    ध्रुव int64_t qk = 32;
    ध्रुव int64_t स्तम्भसङ्ख्या_2 = k_खण्डानि * qk;

    स्वचालित packed_shape_4d = const_i64({n_expert, पङ्क्तिकाः, k_खण्डानि, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, असत्य);

    स्वचालित activations_shape_4d = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(activations, ov::element::i64);
    स्वचालित ids_shape_4d = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(ids, ov::element::i64);
    स्वचालित activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    स्वचालित ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, असत्य);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, असत्य);
    यदि (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    स्वचालित gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    स्थिर ध्रुव std::vector<भिन्न> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<भिन्न> e8m0_lut(256);
    प्रति (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t द्वयङ्काः = static_cast<uint32_t>(i) << 23;
        स्मृतिं_प्रतिलिख(&e8m0_lut[i], &द्वयङ्काः, परिमाण(भिन्न));
    }
    e8m0_lut[0] = std::numeric_limits<भिन्न>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<भिन्न>::quiet_NaN();

    स्वचालित f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    स्वचालित scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    स्वचालित selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    स्वचालित scale_byte = खण्डिका_axis(selected_packed_weights, 4, 0, 1);
    स्वचालित qs = खण्डिका_axis(selected_packed_weights, 4, 1, 17);
    स्वचालित नीचः = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::Auto_प्रसारणम्_Type::NUMPY);
    स्वचालित उच्चः_सरणम् = std::make_shared<ov::op::v15::Bitwise_दक्षिणः_सरणम्>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::Auto_प्रसारणम्_Type::NUMPY);
    स्वचालित nibbles = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{नीचः, उच्चः_सरणम्}, 4);
    स्वचालित nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    स्वचालित weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    स्वचालित scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    स्वचालित scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::Auto_प्रसारणम्_Type::NUMPY);

    स्वचालित ids_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(ids, ov::element::i64);
    स्वचालित selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_सदिशम्{get_dimensions(ids_shape, {0, 1}), const_i64({पङ्क्तिकाः, स्तम्भसङ्ख्या_2})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, असत्य);

    स्वचालित activations_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_सदिशम्{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::प्रसारणम्>(activations, acts_target_dims, ov::op::प्रसारणम्_Type::BIDIRECTIONAL);

    स्वचालित activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> परिणाम =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, असत्य, सत्य);

    स्वचालित batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वचालित row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {पङ्क्तिकाः});
    स्वचालित परिणाम_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_सदिशम्{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim}, 0);
    परिणाम = std::make_shared<ov::op::v1::Reshape>(परिणाम, परिणाम_target_dims, असत्य);

    ध्रुव स्वचालित output_type = context.get_output_type();
    यदि (परिणाम.get_element_type() != output_type) {
        परिणाम = std::make_shared<ov::op::v0::Convert>(परिणाम, output_type);
    }
    प्रत्यागम परिणाम;
}

}  // namespace

Output_सदिशम् translate_mul_mat_id(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 3, 3);

    स्वचालित expert_weights = process_view_निवेशः_new(context, 0);
    स्वचालित activations = process_view_निवेशः_new(context, 1);
    स्वचालित ids = process_view_निवेशः_new(context, 2);

    यदि (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    यदि (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    यदि (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_दीर्घता() == 5) {
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
    स्वचालित expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_अन्तः_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    यदि (expert_weights_rank.get_दीर्घता() == 4) {
        स्वचालित expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, असत्य);
    }

    स्वचालित activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    स्वचालित ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, असत्य);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, असत्य);

    यदि (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ध्रुव स्वचालित output_type = context.get_output_type();
    ध्रुव स्वचालित activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    यदि (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    स्वचालित activations_transpose_क्रमः = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_क्रमः);

    ov::Output<ov::Node> परिणाम = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    स्वचालित परिणाम_transpose_क्रमः = const_i64({1, 0, 2});
    परिणाम = std::make_shared<ov::op::v1::Transpose>(परिणाम, परिणाम_transpose_क्रमः);
    यदि (!context.is_stateful()) {
        स्वचालित unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        परिणाम = std::make_shared<ov::op::v0::Unsqueeze>(परिणाम, unsqueeze_axes);
    }

    यदि (परिणाम.get_element_type() != output_type) {
        परिणाम = std::make_shared<ov::op::v0::Convert>(परिणाम, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({परिणाम}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
