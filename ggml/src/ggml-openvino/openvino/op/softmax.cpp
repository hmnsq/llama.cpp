#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cmath>
#समावेश <cstdint>
#समावेश <cstring>
#समावेश <memory>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/frontend/exception.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/softmax.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

अटल bool is_static_one(स्थिर ov::Dimension & dim) {
    वापस dim.is_static() && dim.get_लंबाई() == 1;
}

अटल bool same_static_dim(स्थिर ov::Dimension & lhs, स्थिर ov::Dimension & rhs) {
    वापस lhs.is_static() && rhs.is_static() && lhs.get_लंबाई() == rhs.get_लंबाई();
}

अटल bool is_attention_sinks_निवेश_shape(स्थिर ov::PartialShape & candidate, स्थिर ov::PartialShape & logits_shape) {
    अगर (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_लंबाई() != 4 ||
        logits_shape.rank().get_लंबाई() != 4) {
        वापस false;
    }

    वापस is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 3);

    भग्न scale = 1.0f;
    भग्न max_bias = 0.0f;
    स_नकल(&scale, (भग्न *) context.get_output_op_params() + 0, माप(भग्न));
    स_नकल(&max_bias, (भग्न *) context.get_output_op_params() + 1, माप(भग्न));

    ov::Output<ov::Node> logits = context.get_निवेश(0);
    स्थिर bool second_निवेश_is_sinks =
        context.get_निवेश_आकार() == 2 && is_attention_sinks_निवेश_shape(context.get_निवेश_shape(1), context.get_output_shape());
    स्थिर bool has_mask = context.get_निवेश_आकार() > 1 && !second_निवेश_is_sinks;
    स्थिर bool has_sinks = second_निवेश_is_sinks || context.get_निवेश_आकार() > 2;
    स्थिर माप_प्रकार sinks_निवेश_idx = second_निवेश_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    अगर (scale != 1.0f) {
        स्वतः scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<भग्न>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    अगर (has_mask) {
        ov::Output<ov::Node> mask = context.get_निवेश(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        अगर (context.get_निवेश_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        अगर (context.has_निवेश(mask_name)) {
            mask = context.get_निवेश(mask_name);
        }

        अगर (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        अगर (max_bias > 0.0f) {
            स्वतः out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            स्थिर uint32_t n_शीर्ष = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_शीर्ष > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            स्थिर uint32_t n_शीर्ष_log2 = 1u << static_cast<uint32_t>(std::न्यूनमान(std::log2(static_cast<भग्न>(n_शीर्ष))));
            स्थिर भग्न m0 = std::घात(2.0f, -(max_bias) / static_cast<भग्न>(n_शीर्ष_log2));
            स्थिर भग्न m1 = std::घात(2.0f, -(max_bias / 2.0f) / static_cast<भग्न>(n_शीर्ष_log2));

            std::vector<भग्न> slopes(n_शीर्ष);
            क्रम (uint32_t h = 0; h < n_शीर्ष; ++h) {
                slopes[h] = h < n_शीर्ष_log2 ? std::घात(m0, static_cast<भग्न>(h + 1)) :
                                              std::घात(m1, static_cast<भग्न>(2 * (h - n_शीर्ष_log2) + 1));
            }

            ov::Output<ov::Node> slope_गाँठ =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_शीर्ष}, slopes);
            अगर (slope_गाँठ.get_element_type() != mask.get_element_type()) {
                slope_गाँठ = std::make_shared<ov::op::v0::Convert>(slope_गाँठ, mask.get_element_type());
            }

            स्वतः slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_शीर्ष), 1, 1});
            स्वतः slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_गाँठ, slope_shape, false);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_निवेश = logits;
    अगर (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_निवेश(sinks_निवेश_idx);
        अगर (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        स्वतः sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        स्वतः sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, false);

        स्वतः logits_shape = std::make_shared<ov::op::v3::ShapeOf>(logits, ov::element::i64);
        स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        स्वतः four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        स्वतः shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        स्वतः sink_prefix_shape = std::make_shared<ov::op::v8::Slice>(logits_shape, zero, three, one, shape_axis);
        स्वतः sink_last_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः sink_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{sink_prefix_shape, sink_last_dim}, 0);
        स्वतः sink_column = std::make_shared<ov::op::v3::Broadcast>(sinks_4d, sink_broadcast_shape,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        softmax_निवेश = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        स्वतः softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_निवेश, -1);
        स्वतः original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        स्वतः res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        वापस rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    स्वतः res = std::make_shared<ov::op::v8::Softmax>(softmax_निवेश, -1);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
