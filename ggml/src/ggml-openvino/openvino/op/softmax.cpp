#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cmath>
#شامل <cstdint>
#شامل <cstring>
#شامل <memory>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/softmax.hpp>
#شامل <vector>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

ایستا منطقی is_static_one(ثابت ov::Dimension & dim) {
    بازگشت dim.is_static() && dim.get_length() == 1;
}

ایستا منطقی same_static_dim(ثابت ov::Dimension & lhs, ثابت ov::Dimension & rhs) {
    بازگشت lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

ایستا منطقی is_attention_sinks_درون_داد_shape(ثابت ov::PartialShape & candidate, ثابت ov::PartialShape & logits_shape) {
    اگر (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        بازگشت نادرست;
    }

    بازگشت is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 3);

    اعشاری scale = 1.0f;
    اعشاری max_bias = 0.0f;
    memcpy(&scale, (اعشاری *) context.get_output_op_params() + 0, اندازه(اعشاری));
    memcpy(&max_bias, (اعشاری *) context.get_output_op_params() + 1, اندازه(اعشاری));

    ov::Output<ov::Node> logits = context.get_درون_داد(0);
    ثابت منطقی second_درون_داد_is_sinks =
        context.get_درون_داد_size() == 2 && is_attention_sinks_درون_داد_shape(context.get_درون_داد_shape(1), context.get_output_shape());
    ثابت منطقی has_mask = context.get_درون_داد_size() > 1 && !second_درون_داد_is_sinks;
    ثابت منطقی has_sinks = second_درون_داد_is_sinks || context.get_درون_داد_size() > 2;
    ثابت size_t sinks_درون_داد_idx = second_درون_داد_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    اگر (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<اعشاری>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    اگر (has_mask) {
        ov::Output<ov::Node> mask = context.get_درون_داد(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        اگر (context.get_درون_داد_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        اگر (context.has_درون_داد(mask_name)) {
            mask = context.get_درون_داد(mask_name);
        }

        اگر (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        اگر (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ثابت uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ثابت uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<اعشاری>(n_head))));
            ثابت اعشاری m0 = std::pow(2.0f, -(max_bias) / static_cast<اعشاری>(n_head_log2));
            ثابت اعشاری m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<اعشاری>(n_head_log2));

            std::vector<اعشاری> slopes(n_head);
            برای (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<اعشاری>(h + 1)) :
                                              std::pow(m1, static_cast<اعشاری>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_گره =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            اگر (slope_گره.get_element_type() != mask.get_element_type()) {
                slope_گره = std::make_shared<ov::op::v0::Convert>(slope_گره, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_گره, slope_shape, نادرست);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_درون_داد = logits;
    اگر (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_درون_داد(sinks_درون_داد_idx);
        اگر (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, نادرست);

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
        softmax_درون_داد = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_درون_داد, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        بازگشت rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_درون_داد, -1);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
