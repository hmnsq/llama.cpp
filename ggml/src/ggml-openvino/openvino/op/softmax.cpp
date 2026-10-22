#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstring>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/softmax.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

स्थिर तार्किक is_static_one(ध्रुव ov::Dimension & dim) {
    पटिनिवत्त dim.is_static() && dim.get_length() == 1;
}

स्थिर तार्किक same_static_dim(ध्रुव ov::Dimension & lhs, ध्रुव ov::Dimension & rhs) {
    पटिनिवत्त lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

स्थिर तार्किक is_attention_sinks_निवेशः_shape(ध्रुव ov::PartialShape & candidate, ध्रुव ov::PartialShape & logits_shape) {
    सचे (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        पटिनिवत्त मिच्छा;
    }

    पटिनिवत्त is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 3);

    भिन्न scale = 1.0f;
    भिन्न max_bias = 0.0f;
    memcpy(&scale, (भिन्न *) context.get_output_op_params() + 0, परिमाण(भिन्न));
    memcpy(&max_bias, (भिन्न *) context.get_output_op_params() + 1, परिमाण(भिन्न));

    ov::Output<ov::Node> logits = context.get_निवेशः(0);
    ध्रुव तार्किक second_निवेशः_is_sinks =
        context.get_निवेशः_size() == 2 && is_attention_sinks_निवेशः_shape(context.get_निवेशः_shape(1), context.get_output_shape());
    ध्रुव तार्किक has_mask = context.get_निवेशः_size() > 1 && !second_निवेशः_is_sinks;
    ध्रुव तार्किक has_sinks = second_निवेशः_is_sinks || context.get_निवेशः_size() > 2;
    ध्रुव size_t sinks_निवेशः_idx = second_निवेशः_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    सचे (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<भिन्न>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    सचे (has_mask) {
        ov::Output<ov::Node> mask = context.get_निवेशः(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        सचे (context.get_निवेशः_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        सचे (context.has_निवेशः(mask_name)) {
            mask = context.get_निवेशः(mask_name);
        }

        सचे (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        सचे (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ध्रुव uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ध्रुव uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<भिन्न>(n_head))));
            ध्रुव भिन्न m0 = std::pow(2.0f, -(max_bias) / static_cast<भिन्न>(n_head_log2));
            ध्रुव भिन्न m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<भिन्न>(n_head_log2));

            std::vector<भिन्न> slopes(n_head);
            अत्थाय (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<भिन्न>(h + 1)) :
                                              std::pow(m1, static_cast<भिन्न>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_ग्रन्थिः =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            सचे (slope_ग्रन्थिः.get_element_type() != mask.get_element_type()) {
                slope_ग्रन्थिः = std::make_shared<ov::op::v0::Convert>(slope_ग्रन्थिः, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_ग्रन्थिः, slope_shape, मिच्छा);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_निवेशः = logits;
    सचे (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_निवेशः(sinks_निवेशः_idx);
        सचे (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, मिच्छा);

        auto logits_shape = std::make_shared<ov::op::v3::ShapeOf>(logits, ov::element::i64);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        auto four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        auto shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        auto sink_prefix_shape = std::make_shared<ov::op::v8::Slice>(logits_shape, zero, three, one, shape_axis);
        auto sink_last_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto sink_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{sink_prefix_shape, sink_last_dim}, 0);
        auto sink_column = std::make_shared<ov::op::v3::Broadcast>(sinks_4d, sink_broadcast_shape,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        softmax_निवेशः = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_निवेशः, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        पटिनिवत्त rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_निवेशः, -1);

    पटिनिवत्त rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
