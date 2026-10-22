#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/exception.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/softmax.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

ਸਥਿਰ ਤਾਰਕਿਕ is_static_one(ਅਚਲ ov::Dimension & dim) {
    ਵਾਪਸ dim.is_static() && dim.get_ਲੰਮਾਈ() == 1;
}

ਸਥਿਰ ਤਾਰਕਿਕ same_static_dim(ਅਚਲ ov::Dimension & lhs, ਅਚਲ ov::Dimension & rhs) {
    ਵਾਪਸ lhs.is_static() && rhs.is_static() && lhs.get_ਲੰਮਾਈ() == rhs.get_ਲੰਮਾਈ();
}

ਸਥਿਰ ਤਾਰਕਿਕ is_attention_sinks_ਨਿਵੇਸ਼_shape(ਅਚਲ ov::PartialShape & candidate, ਅਚਲ ov::PartialShape & logits_shape) {
    ਜੇ (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_ਲੰਮਾਈ() != 4 ||
        logits_shape.rank().get_ਲੰਮਾਈ() != 4) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
Output_ਸਦਿਸ਼ translate_soft_max(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 3);

    ਦਸ਼ਮਲਵ scale = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਦਸ਼ਮਲਵ *) context.get_output_op_params() + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਦਸ਼ਮਲਵ *) context.get_output_op_params() + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ov::Output<ov::Node> logits = context.get_ਨਿਵੇਸ਼(0);
    ਅਚਲ ਤਾਰਕਿਕ ਦੂਜਾ_ਨਿਵੇਸ਼_is_sinks =
        context.get_ਨਿਵੇਸ਼_ਮਾਪ() == 2 && is_attention_sinks_ਨਿਵੇਸ਼_shape(context.get_ਨਿਵੇਸ਼_shape(1), context.get_output_shape());
    ਅਚਲ ਤਾਰਕਿਕ has_ਨਕਾਬ = context.get_ਨਿਵੇਸ਼_ਮਾਪ() > 1 && !ਦੂਜਾ_ਨਿਵੇਸ਼_is_sinks;
    ਅਚਲ ਤਾਰਕਿਕ has_sinks = ਦੂਜਾ_ਨਿਵੇਸ਼_is_sinks || context.get_ਨਿਵੇਸ਼_ਮਾਪ() > 2;
    ਅਚਲ size_t sinks_ਨਿਵੇਸ਼_idx = ਦੂਜਾ_ਨਿਵੇਸ਼_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    ਜੇ (scale != 1.0f) {
        ਸਵੈਚਾਲਿਤ scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ਦਸ਼ਮਲਵ>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_ਨਕਾਬ),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    ਜੇ (has_ਨਕਾਬ) {
        ov::Output<ov::Node> ਨਕਾਬ = context.get_ਨਿਵੇਸ਼(1);

        // For stateful
        std::string ਨਕਾਬ_name = "KQ_mask_sliced";
        ਜੇ (context.get_ਨਿਵੇਸ਼_ਨਾਂ()[1].find("swa") != std::string::npos) {
            ਨਕਾਬ_name = "KQ_mask_swa_sliced";
        }
        ਜੇ (context.has_ਨਿਵੇਸ਼(ਨਕਾਬ_name)) {
            ਨਕਾਬ = context.get_ਨਿਵੇਸ਼(ਨਕਾਬ_name);
        }

        ਜੇ (ਨਕਾਬ.get_element_type() != logits.get_element_type()) {
            ਨਕਾਬ = std::make_shared<ov::op::v0::Convert>(ਨਕਾਬ, logits.get_element_type());
        }

        ਜੇ (max_bias > 0.0f) {
            ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();
            FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(ਬਾਹਰਲਾ_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ਅਚਲ uint32_t n_ਸਿਰ = static_cast<uint32_t>(ਬਾਹਰਲਾ_shape[1]);
            FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(n_ਸਿਰ > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<ਦਸ਼ਮਲਵ>(n_ਸਿਰ))));
            ਅਚਲ ਦਸ਼ਮਲਵ m0 = std::ਘਾਤ(2.0f, -(max_bias) / static_cast<ਦਸ਼ਮਲਵ>(n_ਸਿਰ_log2));
            ਅਚਲ ਦਸ਼ਮਲਵ m1 = std::ਘਾਤ(2.0f, -(max_bias / 2.0f) / static_cast<ਦਸ਼ਮਲਵ>(n_ਸਿਰ_log2));

            std::vector<ਦਸ਼ਮਲਵ> slopes(n_ਸਿਰ);
            ਲਈ (uint32_t h = 0; h < n_ਸਿਰ; ++h) {
                slopes[h] = h < n_ਸਿਰ_log2 ? std::ਘਾਤ(m0, static_cast<ਦਸ਼ਮਲਵ>(h + 1)) :
                                              std::ਘਾਤ(m1, static_cast<ਦਸ਼ਮਲਵ>(2 * (h - n_ਸਿਰ_log2) + 1));
            }

            ov::Output<ov::Node> ਢਲਾਨ_ਗੰਢ =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_ਸਿਰ}, slopes);
            ਜੇ (ਢਲਾਨ_ਗੰਢ.get_element_type() != ਨਕਾਬ.get_element_type()) {
                ਢਲਾਨ_ਗੰਢ = std::make_shared<ov::op::v0::Convert>(ਢਲਾਨ_ਗੰਢ, ਨਕਾਬ.get_element_type());
            }

            ਸਵੈਚਾਲਿਤ ਢਲਾਨ_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_ਸਿਰ), 1, 1});
            ਸਵੈਚਾਲਿਤ ਢਲਾਨ_4d = std::make_shared<ov::op::v1::Reshape>(ਢਲਾਨ_ਗੰਢ, ਢਲਾਨ_shape, ਝੂਠ);
            ਨਕਾਬ = std::make_shared<ov::op::v1::Multiply>(ਨਕਾਬ, ਢਲਾਨ_4d);
        }

        logits = std::make_shared<ov::op::v1::ਜੋੜੋ>(logits, ਨਕਾਬ);
    }

    ov::Output<ov::Node> softmax_ਨਿਵੇਸ਼ = logits;
    ਜੇ (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_ਨਿਵੇਸ਼(sinks_ਨਿਵੇਸ਼_idx);
        ਜੇ (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        ਸਵੈਚਾਲਿਤ sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        ਸਵੈਚਾਲਿਤ sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, ਝੂਠ);

        ਸਵੈਚਾਲਿਤ logits_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(logits, ov::element::i64);
        ਸਵੈਚਾਲਿਤ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        ਸਵੈਚਾਲਿਤ four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        ਸਵੈਚਾਲਿਤ shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        ਸਵੈਚਾਲਿਤ sink_ਅਗੇਤਰ_shape = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(logits_shape, zero, three, one, shape_axis);
        ਸਵੈਚਾਲਿਤ sink_ਆਖਰੀ_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ sink_ਪ੍ਰਸਾਰਣ_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_ਸਦਿਸ਼{sink_ਅਗੇਤਰ_shape, sink_ਆਖਰੀ_dim}, 0);
        ਸਵੈਚਾਲਿਤ sink_ਕਾਲਮ = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(sinks_4d, sink_ਪ੍ਰਸਾਰਣ_shape,
                                                                   ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);
        softmax_ਨਿਵੇਸ਼ = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{logits, sink_ਕਾਲਮ}, 3);

        ਸਵੈਚਾਲਿਤ softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_ਨਿਵੇਸ਼, -1);
        ਸਵੈਚਾਲਿਤ original_ਆਖਰੀ_dim = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(logits_shape, three, four, one, shape_axis);
        ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(softmax_with_sink, zero, original_ਆਖਰੀ_dim, one, three);

        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v8::Softmax>(softmax_ਨਿਵੇਸ਼, -1);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
