#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "gather_matmul.hpp"
#ضمن "ggml-openvino/ggml-openvino-extra.h"

#ضمن <cstdint>
#ضمن <cstring>
#ضمن <limits>
#ضمن <memory>
#ضمن <openvino/op/bitwise_and.hpp>
#ضمن <openvino/op/bitwise_right_shift.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

نطاق {

std::shared_ptr<ov::op::v0::Constant> const_i64(ثابت std::vector<int64_t> & اقدار) {
    أرجع ov::op::v0::Constant::create(ov::element::i64, ov::Shape{اقدار.size()}, اقدار);
}

ov::Output<ov::Node> ٹکڑا_axis(ثابت ov::Output<ov::Node> & داخلہ, int64_t axis, int64_t begin, int64_t end) {
    أرجع std::make_shared<ov::op::v8::ٹکڑا>(داخلہ, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<صحيح> trailing_dims(ثابت ov::Output<ov::Node> & داخلہ, صحيح count) {
    ثابت خودکار rank = داخلہ.get_partial_shape().rank();
    FRONT_سرا_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ثابت صحيح rank_len = static_cast<صحيح>(rank.get_لمائی());
    FRONT_سرا_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<صحيح> dims;
    لكل (صحيح i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    أرجع dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ثابت ov::Output<ov::Node> & داخلہ,
                                                  ثابت std::vector<صحيح> & dims) {
    ثابت خودکار & partial_shape = داخلہ.get_partial_shape();
    إذا (partial_shape.is_static()) {
        std::vector<int64_t> اقدار;
        اقدار.reserve(dims.size());
        لكل (ثابت int64_t dim : dims) {
            اقدار.push_back(partial_shape[dim].get_لمائی());
        }
        أرجع const_i64(اقدار);
    }

    خودکار shape = std::make_shared<ov::op::v3::Shape_دا>(داخلہ, ov::element::i64);
    أرجع get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ثابت گرہ_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    خودکار packed_shape = expert_weights.get_partial_shape().منزل_shape();
    FRONT_سرا_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ثابت int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ثابت int64_t صفاں = static_cast<int64_t>(packed_shape[2]);
    ثابت int64_t k_بلاکاں = static_cast<int64_t>(packed_shape[3]);
    ثابت int64_t qk = 32;
    ثابت int64_t خانے_گنتی_2 = k_بلاکاں * qk;

    خودکار packed_shape_4d = const_i64({n_expert, صفاں, k_بلاکاں, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, خطأ);

    خودکار activations_shape_4d = std::make_shared<ov::op::v3::Shape_دا>(activations, ov::element::i64);
    خودکار ids_shape_4d = std::make_shared<ov::op::v3::Shape_دا>(ids, ov::element::i64);
    خودکار activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    خودکار ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, خطأ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, خطأ);
    إذا (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    ساكن ثابت std::vector<عائم> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<عائم> e8m0_lut(256);
    لكل (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t بٹ = static_cast<uint32_t>(i) << 23;
        حافظہ_نقل(&e8m0_lut[i], &بٹ, حجم(عائم));
    }
    e8m0_lut[0] = std::numeric_limits<عائم>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<عائم>::quiet_NaN();

    خودکار f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    خودکار scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    خودکار selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    خودکار scale_byte = ٹکڑا_axis(selected_packed_weights, 4, 0, 1);
    خودکار qs = ٹکڑا_axis(selected_packed_weights, 4, 1, 17);
    خودکار کم = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::Auto_نشر_Type::NUMPY);
    خودکار بلند_کھسکاؤ = std::make_shared<ov::op::v15::Bitwise_دایاں_کھسکاؤ>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::Auto_نشر_Type::NUMPY);
    خودکار nibbles = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{کم, بلند_کھسکاؤ}, 4);
    خودکار nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    خودکار weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    خودکار scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    خودکار scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::Auto_نشر_Type::NUMPY);

    خودکار ids_shape = std::make_shared<ov::op::v3::Shape_دا>(ids, ov::element::i64);
    خودکار selected_weights_نشانہ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_سمتیہ{get_dimensions(ids_shape, {0, 1}), const_i64({صفاں, خانے_گنتی_2})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_نشانہ_dims, خطأ);

    خودکار activations_shape = std::make_shared<ov::op::v3::Shape_دا>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_نشانہ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_سمتیہ{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::نشر>(activations, acts_نشانہ_dims, ov::op::نشر_Type::BIDIRECTIONAL);

    خودکار activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> نتیجہ =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, خطأ, سچ);

    خودکار batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودکار صف_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {صفاں});
    خودکار نتیجہ_نشانہ_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_سمتیہ{batch_dim, get_dimensions(ids_shape, {0, 1}), صف_dim}, 0);
    نتیجہ = std::make_shared<ov::op::v1::Reshape>(نتیجہ, نتیجہ_نشانہ_dims, خطأ);

    ثابت خودکار output_type = context.get_output_type();
    إذا (نتیجہ.get_element_type() != output_type) {
        نتیجہ = std::make_shared<ov::op::v0::Convert>(نتیجہ, output_type);
    }
    أرجع نتیجہ;
}

}  // namespace

Output_سمتیہ translate_mul_mat_id(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 3, 3);

    خودکار expert_weights = process_view_داخلہ_new(context, 0);
    خودکار activations = process_view_داخلہ_new(context, 1);
    خودکار ids = process_view_داخلہ_new(context, 2);

    إذا (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    إذا (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    إذا (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_لمائی() == 5) {
        أرجع rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    خودکار expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_سرا_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    إذا (expert_weights_rank.get_لمائی() == 4) {
        خودکار expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, خطأ);
    }

    خودکار activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    خودکار ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, خطأ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, خطأ);

    إذا (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ثابت خودکار output_type = context.get_output_type();
    ثابت خودکار activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    إذا (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    خودکار activations_transpose_ترتیب = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_ترتیب);

    ov::Output<ov::Node> نتیجہ = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    خودکار نتیجہ_transpose_ترتیب = const_i64({1, 0, 2});
    نتیجہ = std::make_shared<ov::op::v1::Transpose>(نتیجہ, نتیجہ_transpose_ترتیب);
    إذا (!context.is_stateful()) {
        خودکار unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        نتیجہ = std::make_shared<ov::op::v0::Unsqueeze>(نتیجہ, unsqueeze_axes);
    }

    إذا (نتیجہ.get_element_type() != output_type) {
        نتیجہ = std::make_shared<ov::op::v0::Convert>(نتیجہ, output_type);
    }

    أرجع rename_outputs_with_suffix({نتیجہ}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
