#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/softmax.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_static_one(ಧ್ರುವ ov::Dimension & dim) {
    ಪ್ರತ್ಯಾಗಮ dim.is_static() && dim.get_length() == 1;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ same_static_dim(ಧ್ರುವ ov::Dimension & lhs, ಧ್ರುವ ov::Dimension & rhs) {
    ಪ್ರತ್ಯಾಗಮ lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_attention_sinks_ಒಳಹರಿವು_shape(ಧ್ರುವ ov::PartialShape & candidate, ಧ್ರುವ ov::PartialShape & logits_shape) {
    ಯದಿ (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 3);

    ಭಿನ್ನ scale = 1.0f;
    ಭಿನ್ನ max_bias = 0.0f;
    memcpy(&scale, (ಭಿನ್ನ *) context.get_output_op_params() + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_bias, (ಭಿನ್ನ *) context.get_output_op_params() + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ov::Output<ov::Node> logits = context.get_ಒಳಹರಿವು(0);
    ಧ್ರುವ ತಾರ್ಕಿಕ second_ಒಳಹರಿವು_is_sinks =
        context.get_ಒಳಹರಿವು_size() == 2 && is_attention_sinks_ಒಳಹರಿವು_shape(context.get_ಒಳಹರಿವು_shape(1), context.get_output_shape());
    ಧ್ರುವ ತಾರ್ಕಿಕ has_mask = context.get_ಒಳಹರಿವು_size() > 1 && !second_ಒಳಹರಿವು_is_sinks;
    ಧ್ರುವ ತಾರ್ಕಿಕ has_sinks = second_ಒಳಹರಿವು_is_sinks || context.get_ಒಳಹರಿವು_size() > 2;
    ಧ್ರುವ size_t sinks_ಒಳಹರಿವು_idx = second_ಒಳಹರಿವು_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    ಯದಿ (scale != 1.0f) {
        auto scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ಭಿನ್ನ>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    ಯದಿ (has_mask) {
        ov::Output<ov::Node> mask = context.get_ಒಳಹರಿವು(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        ಯದಿ (context.get_ಒಳಹರಿವು_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        ಯದಿ (context.has_ಒಳಹರಿವು(mask_name)) {
            mask = context.get_ಒಳಹರಿವು(mask_name);
        }

        ಯದಿ (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        ಯದಿ (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ಧ್ರುವ uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ಧ್ರುವ uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<ಭಿನ್ನ>(n_head))));
            ಧ್ರುವ ಭಿನ್ನ m0 = std::pow(2.0f, -(max_bias) / static_cast<ಭಿನ್ನ>(n_head_log2));
            ಧ್ರುವ ಭಿನ್ನ m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<ಭಿನ್ನ>(n_head_log2));

            std::vector<ಭಿನ್ನ> slopes(n_head);
            ಪ್ರತಿ (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<ಭಿನ್ನ>(h + 1)) :
                                              std::pow(m1, static_cast<ಭಿನ್ನ>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_ಗಂಟು =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            ಯದಿ (slope_ಗಂಟು.get_element_type() != mask.get_element_type()) {
                slope_ಗಂಟು = std::make_shared<ov::op::v0::Convert>(slope_ಗಂಟು, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_ಗಂಟು, slope_shape, ಅಸತ್ಯ);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_ಒಳಹರಿವು = logits;
    ಯದಿ (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_ಒಳಹರಿವು(sinks_ಒಳಹರಿವು_idx);
        ಯದಿ (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, ಅಸತ್ಯ);

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
        softmax_ಒಳಹರಿವು = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_ಒಳಹರಿವು, -1);
        auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        auto res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    auto res = std::make_shared<ov::op::v8::Softmax>(softmax_ಒಳಹರಿವು, -1);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
