#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/frontend/exception.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/softmax.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

நிலையான தருக்க is_static_one(மாறிலி ov::Dimension & dim) {
    திரும்பு dim.is_static() && dim.get_length() == 1;
}

நிலையான தருக்க same_static_dim(மாறிலி ov::Dimension & lhs, மாறிலி ov::Dimension & rhs) {
    திரும்பு lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

நிலையான தருக்க is_attention_sinks_உள்ளீடு_shape(மாறிலி ov::PartialShape & candidate, மாறிலி ov::PartialShape & logits_shape) {
    எனில் (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        திரும்பு பொய்;
    }

    திரும்பு is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 3);

    தசம scale = 1.0f;
    தசம max_bias = 0.0f;
    memcpy(&scale, (தசம *) context.get_output_op_params() + 0, அளவு(தசம));
    memcpy(&max_bias, (தசம *) context.get_output_op_params() + 1, அளவு(தசம));

    ov::Output<ov::Node> logits = context.get_உள்ளீடு(0);
    மாறிலி தருக்க second_உள்ளீடு_is_sinks =
        context.get_உள்ளீடு_size() == 2 && is_attention_sinks_உள்ளீடு_shape(context.get_உள்ளீடு_shape(1), context.get_output_shape());
    மாறிலி தருக்க has_mask = context.get_உள்ளீடு_size() > 1 && !second_உள்ளீடு_is_sinks;
    மாறிலி தருக்க has_sinks = second_உள்ளீடு_is_sinks || context.get_உள்ளீடு_size() > 2;
    மாறிலி size_t sinks_உள்ளீடு_idx = second_உள்ளீடு_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    எனில் (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<தசம>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    எனில் (has_mask) {
        ov::Output<ov::Node> mask = context.get_உள்ளீடு(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        எனில் (context.get_உள்ளீடு_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        எனில் (context.has_உள்ளீடு(mask_name)) {
            mask = context.get_உள்ளீடு(mask_name);
        }

        எனில் (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        எனில் (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            மாறிலி uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            மாறிலி uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<தசம>(n_head))));
            மாறிலி தசம m0 = std::pow(2.0f, -(max_bias) / static_cast<தசம>(n_head_log2));
            மாறிலி தசம m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<தசம>(n_head_log2));

            std::vector<தசம> slopes(n_head);
            க்கு (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<தசம>(h + 1)) :
                                              std::pow(m1, static_cast<தசம>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_முனை =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            எனில் (slope_முனை.get_element_type() != mask.get_element_type()) {
                slope_முனை = std::make_shared<ov::op::v0::Convert>(slope_முனை, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_முனை, slope_shape, பொய்);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_உள்ளீடு = logits;
    எனில் (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_உள்ளீடு(sinks_உள்ளீடு_idx);
        எனில் (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, பொய்);

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
        softmax_உள்ளீடு = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_உள்ளீடு, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        திரும்பு rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_உள்ளீடு, -1);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
