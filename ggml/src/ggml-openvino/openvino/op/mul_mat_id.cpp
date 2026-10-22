#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"
#అన్తర్భావ "gather_matmul.hpp"
#అన్తర్భావ "ggml-openvino/ggml-openvino-extra.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstring>
#అన్తర్భావ <limits>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/bitwise_and.hpp>
#అన్తర్భావ <openvino/op/bitwise_right_shift.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

నామాకాశ {

std::shared_ptr<ov::op::v0::Constant> const_i64(స్థిరరాశి std::vector<int64_t> & విలువలు) {
    తిరిగి ov::op::v0::Constant::create(ov::element::i64, ov::Shape{విలువలు.size()}, విలువలు);
}

ov::Output<ov::Node> ముక్క_axis(స్థిరరాశి ov::Output<ov::Node> & ఆదానం, int64_t axis, int64_t begin, int64_t end) {
    తిరిగి std::make_shared<ov::op::v8::ముక్క>(ఆదానం, const_i64({begin}), const_i64({end}), const_i64({1}),
                                              const_i64({axis}));
}

// GGML tensors are rank 4, but stateful models drop the leading size-1 batch dim, so
// activations and ids arrive one rank lower. Pick the trailing dims by actual rank.
std::vector<పూర్ణాంకం> trailing_dims(స్థిరరాశి ov::Output<ov::Node> & ఆదానం, పూర్ణాంకం count) {
    స్థిరరాశి స్వయంచాలక rank = ఆదానం.get_partial_shape().rank();
    FRONT_చివర_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MUL_MAT_ID input");
    స్థిరరాశి పూర్ణాంకం rank_len = static_cast<పూర్ణాంకం>(rank.get_నిడివి());
    FRONT_చివర_OP_CONVERSION_CHECK(rank_len >= count, "MUL_MAT_ID input rank is too low");

    std::vector<పూర్ణాంకం> dims;
    కొరకు (పూర్ణాంకం i = rank_len - count; i < rank_len; ++i) {
        dims.push_back(i);
    }
    తిరిగి dims;
}

ov::Output<ov::Node> static_shape_dims_or_shapeof(స్థిరరాశి ov::Output<ov::Node> & ఆదానం,
                                                  స్థిరరాశి std::vector<పూర్ణాంకం> & dims) {
    స్థిరరాశి స్వయంచాలక & partial_shape = ఆదానం.get_partial_shape();
    ఒకవేళ (partial_shape.is_static()) {
        std::vector<int64_t> విలువలు;
        విలువలు.reserve(dims.size());
        కొరకు (స్థిరరాశి int64_t dim : dims) {
            విలువలు.push_back(partial_shape[dim].get_నిడివి());
        }
        తిరిగి const_i64(విలువలు);
    }

    స్వయంచాలక shape = std::make_shared<ov::op::v3::Shape_యొక్క>(ఆదానం, ov::element::i64);
    తిరిగి get_dimensions(shape, dims);
}

ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(స్థిరరాశి కణుపు_Context & context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    స్వయంచాలక packed_shape = expert_weights.get_partial_shape().గమ్యం_shape();
    FRONT_చివర_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    స్థిరరాశి int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    స్థిరరాశి int64_t అడ్డువరుసలు = static_cast<int64_t>(packed_shape[2]);
    స్థిరరాశి int64_t k_ఖండాలు = static_cast<int64_t>(packed_shape[3]);
    స్థిరరాశి int64_t qk = 32;
    స్థిరరాశి int64_t నిలువుల_సంఖ్య_2 = k_ఖండాలు * qk;

    స్వయంచాలక packed_shape_4d = const_i64({n_expert, అడ్డువరుసలు, k_ఖండాలు, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, అబద్ధం);

    స్వయంచాలక activations_shape_4d = std::make_shared<ov::op::v3::Shape_యొక్క>(activations, ov::element::i64);
    స్వయంచాలక ids_shape_4d = std::make_shared<ov::op::v3::Shape_యొక్క>(ids, ov::element::i64);
    స్వయంచాలక activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    స్వయంచాలక ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, అబద్ధం);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, అబద్ధం);
    ఒకవేళ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    స్వయంచాలక gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    స్థిర స్థిరరాశి std::vector<దశాంశం> f4e2m1_lut = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    std::vector<దశాంశం> e8m0_lut(256);
    కొరకు (size_t i = 0; i < e8m0_lut.size(); ++i) {
        uint32_t బిట్లు = static_cast<uint32_t>(i) << 23;
        జ్ఞాపకం_నకలు(&e8m0_lut[i], &బిట్లు, పరిమాణం(దశాంశం));
    }
    e8m0_lut[0] = std::numeric_limits<దశాంశం>::min() / 2.0f;
    e8m0_lut[255] = std::numeric_limits<దశాంశం>::quiet_NaN();

    స్వయంచాలక f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    స్వయంచాలక scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    స్వయంచాలక selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    స్వయంచాలక scale_byte = ముక్క_axis(selected_packed_weights, 4, 0, 1);
    స్వయంచాలక qs = ముక్క_axis(selected_packed_weights, 4, 1, 17);
    స్వయంచాలక దిగువ = std::make_shared<ov::op::v13::BitwiseAnd>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}), ov::op::Auto_ప్రసారం_Type::NUMPY);
    స్వయంచాలక ఎగువ_జరుపు = std::make_shared<ov::op::v15::Bitwise_కుడి_జరుపు>(
        qs, ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}), ov::op::Auto_ప్రసారం_Type::NUMPY);
    స్వయంచాలక nibbles = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{దిగువ, ఎగువ_జరుపు}, 4);
    స్వయంచాలక nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    స్వయంచాలక weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    స్వయంచాలక scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    స్వయంచాలక scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32,
                                                                                  ov::op::Auto_ప్రసారం_Type::NUMPY);

    స్వయంచాలక ids_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(ids, ov::element::i64);
    స్వయంచాలక selected_weights_లక్ష్యం_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_సదిశ{get_dimensions(ids_shape, {0, 1}), const_i64({అడ్డువరుసలు, నిలువుల_సంఖ్య_2})}, 0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_లక్ష్యం_dims, అబద్ధం);

    స్వయంచాలక activations_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(activations, ov::element::i64);
    ov::Output<ov::Node> acts_లక్ష్యం_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_సదిశ{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted =
        std::make_shared<ov::op::v3::ప్రసారం>(activations, acts_లక్ష్యం_dims, ov::op::ప్రసారం_Type::BIDIRECTIONAL);

    స్వయంచాలక activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> ఫలితం =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, అబద్ధం, నిజం);

    స్వయంచాలక batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    స్వయంచాలక అడ్డువరుస_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {అడ్డువరుసలు});
    స్వయంచాలక ఫలితం_లక్ష్యం_dims = std::make_shared<ov::op::v0::Concat>(
        ov::Output_సదిశ{batch_dim, get_dimensions(ids_shape, {0, 1}), అడ్డువరుస_dim}, 0);
    ఫలితం = std::make_shared<ov::op::v1::Reshape>(ఫలితం, ఫలితం_లక్ష్యం_dims, అబద్ధం);

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (ఫలితం.get_element_type() != output_type) {
        ఫలితం = std::make_shared<ov::op::v0::Convert>(ఫలితం, output_type);
    }
    తిరిగి ఫలితం;
}

}  // namespace

Output_సదిశ translate_mul_mat_id(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 3, 3);

    స్వయంచాలక expert_weights = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక activations = process_view_ఆదానం_new(context, 1);
    స్వయంచాలక ids = process_view_ఆదానం_new(context, 2);

    ఒకవేళ (activations.get_partial_shape().rank() == 3) {
        activations = std::make_shared<ov::op::v0::Unsqueeze>(activations, const_i64({0}));
    }
    ఒకవేళ (ids.get_partial_shape().rank() == 3) {
        ids = std::make_shared<ov::op::v0::Unsqueeze>(ids, const_i64({0}));
    }

    ఒకవేళ (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_నిడివి() == 5) {
        తిరిగి rename_outputs_with_suffix({translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
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
    స్వయంచాలక expert_weights_rank = expert_weights.get_partial_shape().rank();
    FRONT_చివర_OP_CONVERSION_CHECK(expert_weights_rank.is_static(),
                                  "Expected static rank for MUL_MAT_ID expert weights");
    ఒకవేళ (expert_weights_rank.get_నిడివి() == 4) {
        స్వయంచాలక expert_weights_shape_3d = static_shape_dims_or_shapeof(expert_weights, {1, 2, 3});
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape_3d, అబద్ధం);
    }

    స్వయంచాలక activations_shape_3d = static_shape_dims_or_shapeof(activations, trailing_dims(activations, 3));
    స్వయంచాలక ids_shape_2d = static_shape_dims_or_shapeof(ids, trailing_dims(ids, 2));

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, అబద్ధం);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, అబద్ధం);

    ఒకవేళ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    స్థిరరాశి స్వయంచాలక activations_type = ggml_openvino_is_gpu() ? ov::element::f16 : ov::element::f32;
    ఒకవేళ (activations.get_element_type() != activations_type) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, activations_type);
    }

    // GatherMatmul's A input is [n_used_or_1, n_tokens, k]; activations_3d is
    // [n_tokens, n_used_or_1, k].
    స్వయంచాలక activations_transpose_క్రమం = const_i64({1, 0, 2});
    ov::Output<ov::Node> activations_for_gather =
        std::make_shared<ov::op::v1::Transpose>(activations, activations_transpose_క్రమం);

    ov::Output<ov::Node> ఫలితం = std::make_shared<ov::op::internal::GatherMatmul>(activations_for_gather, expert_weights, ids);

    // result is [n_used, n_tokens, m]; GGML expects [1, n_tokens, n_used, m], except on the
    // stateful path where the leading batch dim is dropped.
    స్వయంచాలక ఫలితం_transpose_క్రమం = const_i64({1, 0, 2});
    ఫలితం = std::make_shared<ov::op::v1::Transpose>(ఫలితం, ఫలితం_transpose_క్రమం);
    ఒకవేళ (!context.is_stateful()) {
        స్వయంచాలక unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ఫలితం = std::make_shared<ov::op::v0::Unsqueeze>(ఫలితం, unsqueeze_axes);
    }

    ఒకవేళ (ఫలితం.get_element_type() != output_type) {
        ఫలితం = std::make_shared<ov::op::v0::Convert>(ఫలితం, output_type);
    }

    తిరిగి rename_outputs_with_suffix({ఫలితం}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
