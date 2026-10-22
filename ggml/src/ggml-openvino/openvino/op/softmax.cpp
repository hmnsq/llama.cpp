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

स्थिर तार्किक is_static_one(स्थिरांक ov::Dimension & dim) {
    परत dim.is_static() && dim.get_length() == 1;
}

स्थिर तार्किक same_static_dim(स्थिरांक ov::Dimension & lhs, स्थिरांक ov::Dimension & rhs) {
    परत lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

स्थिर तार्किक is_attention_sinks_आदान_shape(स्थिरांक ov::PartialShape & candidate, स्थिरांक ov::PartialShape & logits_shape) {
    जर (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        परत असत्य;
    }

    परत is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
OutputVector translate_soft_max(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 3);

    दशांश scale = 1.0f;
    दशांश max_bias = 0.0f;
    स_नकल(&scale, (दशांश *) context.get_output_op_params() + 0, आकार(दशांश));
    स_नकल(&max_bias, (दशांश *) context.get_output_op_params() + 1, आकार(दशांश));

    ov::Output<ov::Node> logits = context.get_आदान(0);
    स्थिरांक तार्किक second_आदान_is_sinks =
        context.get_आदान_size() == 2 && is_attention_sinks_आदान_shape(context.get_आदान_shape(1), context.get_output_shape());
    स्थिरांक तार्किक has_mask = context.get_आदान_size() > 1 && !second_आदान_is_sinks;
    स्थिरांक तार्किक has_sinks = second_आदान_is_sinks || context.get_आदान_size() > 2;
    स्थिरांक माप_प्रकार sinks_आदान_idx = second_आदान_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    जर (scale != 1.0f) {
        स्वतः scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<दशांश>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_mask),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    जर (has_mask) {
        ov::Output<ov::Node> mask = context.get_आदान(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        जर (context.get_आदान_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        जर (context.has_आदान(mask_name)) {
            mask = context.get_आदान(mask_name);
        }

        जर (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        जर (max_bias > 0.0f) {
            स्वतः out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            स्थिरांक uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            स्थिरांक uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::न्यूनमान(std::log2(static_cast<दशांश>(n_head))));
            स्थिरांक दशांश m0 = std::घात(2.0f, -(max_bias) / static_cast<दशांश>(n_head_log2));
            स्थिरांक दशांश m1 = std::घात(2.0f, -(max_bias / 2.0f) / static_cast<दशांश>(n_head_log2));

            std::vector<दशांश> slopes(n_head);
            साठी (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::घात(m0, static_cast<दशांश>(h + 1)) :
                                              std::घात(m1, static_cast<दशांश>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_गाठ =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            जर (slope_गाठ.get_element_type() != mask.get_element_type()) {
                slope_गाठ = std::make_shared<ov::op::v0::Convert>(slope_गाठ, mask.get_element_type());
            }

            स्वतः slope_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            स्वतः slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_गाठ, slope_shape, असत्य);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> softmax_आदान = logits;
    जर (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_आदान(sinks_आदान_idx);
        जर (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        स्वतः sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        स्वतः sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, असत्य);

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
        softmax_आदान = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);

        स्वतः softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_आदान, -1);
        स्वतः original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
        स्वतः res = std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);

        परत rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    स्वतः res = std::make_shared<ov::op::v8::Softmax>(softmax_आदान, -1);

    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
