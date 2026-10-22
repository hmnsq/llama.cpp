#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cmath>
#ضمن <cstdint>
#ضمن <cstring>
#ضمن <memory>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/softmax.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

ساكن منطقي is_static_one(ثابت ov::Dimension & dim) {
    أرجع dim.is_static() && dim.get_length() == 1;
}

ساكن منطقي same_static_dim(ثابت ov::Dimension & lhs, ثابت ov::Dimension & rhs) {
    أرجع lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

ساكن منطقي is_attention_sinks_شغرو_shape(ثابت ov::PartialShape & candidate, ثابت ov::PartialShape & logits_shape) {
    إذا (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        أرجع خطأ;
    }

    أرجع is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(ثابت كلي_Context & context) {
    لمب_inputs_check(context, 1, 3);

    عائم scale = 1.0f;
    عائم max_bias = 0.0f;
    memcpy(&scale, (عائم *) context.get_output_op_params() + 0, حجم(عائم));
    memcpy(&max_bias, (عائم *) context.get_output_op_params() + 1, حجم(عائم));

    ov::Output<ov::Node> logits = context.get_شغرو(0);
    ثابت منطقي second_شغرو_is_sinks =
        context.get_شغرو_size() == 2 && is_attention_sinks_شغرو_shape(context.get_شغرو_shape(1), context.get_output_shape());
    ثابت منطقي has_mask = context.get_شغرو_size() > 1 && !second_شغرو_is_sinks;
    ثابت منطقي has_sinks = second_شغرو_is_sinks || context.get_شغرو_size() > 2;
    ثابت size_t sinks_شغرو_idx = second_شغرو_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    إذا (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<عائم>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    إذا (has_mask) {
        ov::Output<ov::Node> mask = context.get_شغرو(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        إذا (context.get_شغرو_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        إذا (context.has_شغرو(mask_name)) {
            mask = context.get_شغرو(mask_name);
        }

        إذا (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        إذا (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ثابت uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ثابت uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<عائم>(n_head))));
            ثابت عائم m0 = std::pow(2.0f, -(max_bias) / static_cast<عائم>(n_head_log2));
            ثابت عائم m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<عائم>(n_head_log2));

            std::vector<عائم> slopes(n_head);
            لكل (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<عائم>(h + 1)) :
                                              std::pow(m1, static_cast<عائم>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_كلي =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            إذا (slope_كلي.get_element_type() != mask.get_element_type()) {
                slope_كلي = std::make_shared<ov::op::v0::Convert>(slope_كلي, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_كلي, slope_shape, خطأ);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_شغرو = logits;
    إذا (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_شغرو(sinks_شغرو_idx);
        إذا (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, خطأ);

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
        softmax_شغرو = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_شغرو, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_شغرو, -1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
