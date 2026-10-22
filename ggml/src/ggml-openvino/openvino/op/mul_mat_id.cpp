#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "gather_matmul.hpp"
#شامل "ggml-openvino/ggml-openvino-extra.h"

#شامل <cstdint>
#شامل <cstring>
#شامل <limits>
#شامل <memory>
#شامل <openvino/op/bitwise_and.hpp>
#شامل <openvino/op/bitwise_right_shift.hpp>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

نامخلا {

std::shared_ptr<ov::op::v0::Constant> const_i64(ثابت std::vector<int64_t> & اقدار) {
    واپس ov::op::v0::Constant::create(ov::element::i64, ov::Shape{اقدار.size()}, اقدار);
}

ov::Output<ov::Node> قاش_axis(ثابت ov::Output<ov::Node> & داخلہ, int64_t axis, int64_t begin, int64_t end) {
    واپس std::make_shared<ov::op::v8::قاش>(داخلہ, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<صحیح> trailing_dims(ثابت ov::Output<ov::Node> & داخلہ, صحیح count) {
    ثابت خودکار rank = داخلہ.get_partial_shape().rank();
    FRONT_انتہا_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ثابت صحیح rank_len = static_cast<صحیح>(rank.get_لمبائی());
    FRONT_انتہا_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<صحیح> dims;
    برائے (صحیح i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    واپس dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ثابت ov::Output<ov::Node> & داخلہ,
                                                  ثابت std::vector<صحیح> & dims) {
    ثابت خودکار & partial_shape = داخلہ.get_partial_shape();
    اگر (partial_shape.is_static()) {
        std::vector<int64_t> اقدار;
        اقدار.reserve(dims.size());
        برائے (ثابت int64_t dim : dims) {
            اقدار.push_back(partial_shape[dim].get_لمبائی());
        }
        واپس const_i64(اقدار);
    }

    خودکار shape = std::make_shared<ov::op::v3::Shape_کا>(داخلہ, ov::element::i64);
    واپس get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ثابت گرہ_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    خودکار packed_shape = expert_weights.get_partial_shape().مقصد_shape();
    FRONT_انتہا_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ثابت int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ثابت int64_t صفیں = static_cast<int64_t>(packed_shape[2]);
    ثابت int64_t k_بلاکیں = static_cast<int64_t>(packed_shape[3]);
    ثابت int64_t qk = 32;
    ثابت int64_t ستون_تعداد_2 = k_بلاکیں * qk;

    خودکار packed_shape_4d = const_i64({n_expert, صفیں, k_بلاکیں, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, جھوٹ);

    خودکار activations_shape_4d = std::make_shared<ov::op::v3::Shape_کا>(activations, ov::element::i64);
    خودکار ids_shape_4d = std::make_shared<ov::op::v3::Shape_کا>(ids, ov::element::i64);
    خودکار activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    خودکار ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, جھوٹ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, جھوٹ);
    اگر (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    ساکن ثابت std::vector<اعشاری> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<اعشاری> e8m0_lut(256);
    برائے (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t بٹ = static_cast<uint32_t>(i) << 23;
        حافظہ_نقل(&e8m0_lut[i], &بٹ, حجم(اعشاری));
    }
    e8m0_lut[0] = std::numeric_limits<اعشاری>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<اعشاری>::quiet_NaN();

    خودکار f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    خودکار scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    خودکار selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    خودکار scale_byte = قاش_axis(selected_packed_weights, 4, 0, 1);
    خودکار qs = قاش_axis(selected_packed_weights, 4, 1, 17);
    خودکار کم = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::Auto_نشر_Type::NUMPY);
    خودکار بلند_سرکاؤ = std::make_shared<ov::op::v15::Bitwise_دایاں_سرکاؤ>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::Auto_نشر_Type::NUMPY);
    خودکار nibbles = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{کم, بلند_سرکاؤ}, 4);
    خودکار nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    خودکار weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    خودکار scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    خودکار scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::Auto_نشر_Type::NUMPY);

    خودکار ids_shape = std::make_shared<ov::op::v3::Shape_کا>(ids, ov::element::i64);
    خودکار selected_weights_ہدف_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_سمتیہ{get_dimensions(ids_shape, {0, 1}), const_i64({صفیں, ستون_تعداد_2})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_ہدف_dims, جھوٹ);

    خودکار activations_shape = std::make_shared<ov::op::v3::Shape_کا>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_ہدف_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_سمتیہ{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::نشر>(activations, acts_ہدف_dims, ov::op::نشر_Type::BIDIRECTIONAL);

    خودکار activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> نتیجہ =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, جھوٹ, سچ);

    خودکار batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودکار صف_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {صفیں});
    خودکار نتیجہ_ہدف_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_سمتیہ{batch_dim, get_dimensions(ids_shape, {0, 1}), صف_dim}, 0);
    نتیجہ = std::make_shared<ov::op::v1::Reshape>(نتیجہ, نتیجہ_ہدف_dims, جھوٹ);

    ثابت خودکار output_type = context.get_output_type();
    اگر (نتیجہ.get_element_type() != output_type) {
        نتیجہ = std::make_shared<ov::op::v0::Convert>(نتیجہ, output_type);
    }
    واپس نتیجہ;
}

}  // namespace

Output_سمتیہ translate_mul_mat_id(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 3, 3);

    خودکار expert_weights = process_view_داخلہ_new(context, 0);
    خودکار activations = process_view_داخلہ_new(context, 1);
    خودکار ids = process_view_داخلہ_new(context, 2);

    اگر (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    اگر (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    اگر (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_لمبائی() == 5) {
        واپس rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    FRONT_انتہا_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    اگر (expert_weights_rank.get_لمبائی() == 4) {
        خودکار expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, جھوٹ);
    }

    خودکار activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    خودکار ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, جھوٹ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, جھوٹ);

    اگر (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ثابت خودکار output_type = context.get_output_type();
    ثابت خودکار activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    اگر (activations.get_element_type() != activations_type) {
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
    اگر (!context.is_stateful()) {
        خودکار unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        نتیجہ = std::make_shared<ov::op::v0::Unsqueeze>(نتیجہ, unsqueeze_axes);
    }

    اگر (نتیجہ.get_element_type() != output_type) {
        نتیجہ = std::make_shared<ov::op::v0::Convert>(نتیجہ, output_type);
    }

    واپس rename_outputs_with_suffix({نتیجہ}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
