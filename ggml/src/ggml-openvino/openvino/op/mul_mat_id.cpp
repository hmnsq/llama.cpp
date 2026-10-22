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

std::shared_ptr<ov::op::v0::Constant> const_i64(ثابت std::vector<int64_t> & قدرون) {
    أرجع ov::op::v0::Constant::create(ov::element::i64, ov::Shape{قدرون.size()}, قدرون);
}

ov::Output<ov::Node> ٽڪرو_axis(ثابت ov::Output<ov::Node> & داخلا_2, int64_t axis, int64_t begin, int64_t end) {
    أرجع std::make_shared<ov::op::v8::ٽڪرو>(داخلا_2, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<صحيح> trailing_dims(ثابت ov::Output<ov::Node> & داخلا_2, صحيح count) {
    ثابت خودڪار rank = داخلا_2.get_partial_shape().rank();
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    ثابت صحيح rank_len = static_cast<صحيح>(rank.get_ڊيگهائي());
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<صحيح> dims;
    لكل (صحيح i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    أرجع dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(ثابت ov::Output<ov::Node> & داخلا_2,
                                                  ثابت std::vector<صحيح> & dims) {
    ثابت خودڪار & partial_shape = داخلا_2.get_partial_shape();
    إذا (partial_shape.is_static()) {
        std::vector<int64_t> قدرون;
        قدرون.reserve(dims.size());
        لكل (ثابت int64_t dim : dims) {
            قدرون.push_back(partial_shape[dim].get_ڊيگهائي());
        }
        أرجع const_i64(قدرون);
    }

    خودڪار shape = std::make_shared<ov::op::v3::Shape_جو>(داخلا_2, ov::element::i64);
    أرجع get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(ثابت ڳنڍ_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    خودڪار packed_shape = expert_weights.get_partial_shape().منزل_shape();
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    ثابت int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    ثابت int64_t صفون = static_cast<int64_t>(packed_shape[2]);
    ثابت int64_t k_بلاڪون = static_cast<int64_t>(packed_shape[3]);
    ثابت int64_t qk = 32;
    ثابت int64_t کانن_ڳڻپ_2 = k_بلاڪون * qk;

    خودڪار packed_shape_4d = const_i64({n_expert, صفون, k_بلاڪون, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, خطأ);

    خودڪار activations_shape_4d = std::make_shared<ov::op::v3::Shape_جو>(activations, ov::element::i64);
    خودڪار ids_shape_4d = std::make_shared<ov::op::v3::Shape_جو>(ids, ov::element::i64);
    خودڪار activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    خودڪار ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, خطأ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, خطأ);
    إذا (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    خودڪار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    ساكن ثابت std::vector<عائم> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<عائم> e8m0_lut(256);
    لكل (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t بٽ = static_cast<uint32_t>(i) << 23;
        يادگيري_نقل(&e8m0_lut[i], &بٽ, حجم(عائم));
    }
    e8m0_lut[0] = std::numeric_limits<عائم>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<عائم>::quiet_NaN();

    خودڪار f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    خودڪار scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    خودڪار selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    خودڪار scale_byte = ٽڪرو_axis(selected_packed_weights, 4, 0, 1);
    خودڪار qs = ٽڪرو_axis(selected_packed_weights, 4, 1, 17);
    خودڪار هيٺيون = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::Auto_نشر_Type::NUMPY);
    خودڪار مٿاهون_سرڪايو = std::make_shared<ov::op::v15::Bitwise_ساڄو_سرڪايو>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::Auto_نشر_Type::NUMPY);
    خودڪار nibbles = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{هيٺيون, مٿاهون_سرڪايو}, 4);
    خودڪار nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    خودڪار weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    خودڪار scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    خودڪار scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::Auto_نشر_Type::NUMPY);

    خودڪار ids_shape = std::make_shared<ov::op::v3::Shape_جو>(ids, ov::element::i64);
    خودڪار selected_weights_هدف_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ويڪٽر{get_dimensions(ids_shape, {0, 1}), const_i64({صفون, کانن_ڳڻپ_2})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_هدف_dims, خطأ);

    خودڪار activations_shape = std::make_shared<ov::op::v3::Shape_جو>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_هدف_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ويڪٽر{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::نشر>(activations, acts_هدف_dims, ov::op::نشر_Type::BIDIRECTIONAL);

    خودڪار activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> نتيجو =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, خطأ, سچ);

    خودڪار batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودڪار صف_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {صفون});
    خودڪار نتيجو_هدف_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_ويڪٽر{batch_dim, get_dimensions(ids_shape, {0, 1}), صف_dim}, 0);
    نتيجو = std::make_shared<ov::op::v1::Reshape>(نتيجو, نتيجو_هدف_dims, خطأ);

    ثابت خودڪار output_type = context.get_output_type();
    إذا (نتيجو.get_element_type() != output_type) {
        نتيجو = std::make_shared<ov::op::v0::Convert>(نتيجو, output_type);
    }
    أرجع نتيجو;
}

}  // namespace

Output_ويڪٽر translate_mul_mat_id(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 3, 3);

    خودڪار expert_weights = process_view_داخلا_new(context, 0);
    خودڪار activations = process_view_داخلا_new(context, 1);
    خودڪار ids = process_view_داخلا_new(context, 2);

    إذا (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    إذا (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    إذا (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_ڊيگهائي() == 5) {
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
    خودڪار expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    إذا (expert_weights_rank.get_ڊيگهائي() == 4) {
        خودڪار expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, خطأ);
    }

    خودڪار activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    خودڪار ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, خطأ);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, خطأ);

    إذا (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ثابت خودڪار output_type = context.get_output_type();
    ثابت خودڪار activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    إذا (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    خودڪار activations_transpose_ترتيب = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_ترتيب);

    ov::Output<ov::Node> نتيجو = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    خودڪار نتيجو_transpose_ترتيب = const_i64({1, 0, 2});
    نتيجو = std::make_shared<ov::op::v1::Transpose>(نتيجو, نتيجو_transpose_ترتيب);
    إذا (!context.is_stateful()) {
        خودڪار unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        نتيجو = std::make_shared<ov::op::v0::Unsqueeze>(نتيجو, unsqueeze_axes);
    }

    إذا (نتيجو.get_element_type() != output_type) {
        نتيجو = std::make_shared<ov::op::v0::Convert>(نتيجو, output_type);
    }

    أرجع rename_outputs_with_suffix({نتيجو}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
