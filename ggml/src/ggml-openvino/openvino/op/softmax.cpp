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

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

ساکن منطقی is_static_one(ثابت ov::Dimension & dim) {
    واپس dim.is_static() && dim.get_لمبائی() == 1;
}

ساکن منطقی same_static_dim(ثابت ov::Dimension & lhs, ثابت ov::Dimension & rhs) {
    واپس lhs.is_static() && rhs.is_static() && lhs.get_لمبائی() == rhs.get_لمبائی();
}

ساکن منطقی is_attention_sinks_داخلہ_shape(ثابت ov::PartialShape & candidate, ثابت ov::PartialShape & logits_shape) {
    اگر (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_لمبائی() != 4 ||
        logits_shape.rank().get_لمبائی() != 4) {
        واپس جھوٹ;
    }

    واپس is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) append attention sinks as hidden logits (if provided)
// 4) softmax over the last dimension and remove the hidden sink column
Output_سمتیہ translate_soft_max(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 3);

    اعشاری scale = 1.0f;
    اعشاری max_bias = 0.0f;
    حافظہ_نقل(&scale, (اعشاری *) context.get_output_op_params() + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias, (اعشاری *) context.get_output_op_params() + 1, حجم(اعشاری));

    ov::Output<ov::Node> logits = context.get_داخلہ(0);
    ثابت منطقی دوسرا_داخلہ_is_sinks =
        context.get_داخلہ_سائز() == 2 && is_attention_sinks_داخلہ_shape(context.get_داخلہ_shape(1), context.get_output_shape());
    ثابت منطقی has_نقاب = context.get_داخلہ_سائز() > 1 && !دوسرا_داخلہ_is_sinks;
    ثابت منطقی has_sinks = دوسرا_داخلہ_is_sinks || context.get_داخلہ_سائز() > 2;
    ثابت size_t sinks_داخلہ_idx = دوسرا_داخلہ_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    اگر (scale != 1.0f) {
        خودکار scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<اعشاری>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_انتہا_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_نقاب),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    اگر (has_نقاب) {
        ov::Output<ov::Node> نقاب = context.get_داخلہ(1);

        // For stateful
        std::string نقاب_name = "KQ_mask_sliced";
        اگر (context.get_داخلہ_نام()[1].find("swa") != std::string::npos) {
            نقاب_name = "KQ_mask_swa_sliced";
        }
        اگر (context.has_داخلہ(نقاب_name)) {
            نقاب = context.get_داخلہ(نقاب_name);
        }

        اگر (نقاب.get_element_type() != logits.get_element_type()) {
            نقاب = std::make_shared<ov::op::v0::Convert>(نقاب, logits.get_element_type());
        }

        اگر (max_bias > 0.0f) {
            خودکار برآمد_shape = context.get_output_shape().مقصد_shape();
            FRONT_انتہا_CHECK_IMPLEMENTED(برآمد_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ثابت uint32_t n_سر = static_cast<uint32_t>(برآمد_shape[1]);
            FRONT_انتہا_CHECK_IMPLEMENTED(n_سر > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ثابت uint32_t n_سر_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<اعشاری>(n_سر))));
            ثابت اعشاری m0 = std::قوت(2.0f, -(max_bias) / static_cast<اعشاری>(n_سر_log2));
            ثابت اعشاری m1 = std::قوت(2.0f, -(max_bias / 2.0f) / static_cast<اعشاری>(n_سر_log2));

            std::vector<اعشاری> slopes(n_سر);
            برائے (uint32_t h = 0; h < n_سر; ++h) {
                slopes[h] = h < n_سر_log2 ? std::قوت(m0, static_cast<اعشاری>(h + 1)) :
                                              std::قوت(m1, static_cast<اعشاری>(2 * (h - n_سر_log2) + 1));
            }

            ov::Output<ov::Node> ڈھلان_گرہ =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_سر}, slopes);
            اگر (ڈھلان_گرہ.get_element_type() != نقاب.get_element_type()) {
                ڈھلان_گرہ = std::make_shared<ov::op::v0::Convert>(ڈھلان_گرہ, نقاب.get_element_type());
            }

            خودکار ڈھلان_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_سر), 1, 1});
            خودکار ڈھلان_4d = std::make_shared<ov::op::v1::Reshape>(ڈھلان_گرہ, ڈھلان_shape, جھوٹ);
            نقاب = std::make_shared<ov::op::v1::Multiply>(نقاب, ڈھلان_4d);
        }

        logits = std::make_shared<ov::op::v1::جوڑو>(logits, نقاب);
    }

    ov::Output<ov::Node> softmax_داخلہ = logits;
    اگر (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_داخلہ(sinks_داخلہ_idx);
        اگر (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        خودکار sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        خودکار sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, جھوٹ);

        خودکار logits_shape = std::make_shared<ov::op::v3::Shape_کا>(logits, ov::element::i64);
        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        خودکار four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        خودکار shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        خودکار sink_سابقہ_shape = std::make_shared<ov::op::v8::قاش>(logits_shape, zero, three, one, shape_axis);
        خودکار sink_آخری_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار sink_نشر_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_سمتیہ{sink_سابقہ_shape, sink_آخری_dim}, 0);
        خودکار sink_عمود = std::make_shared<ov::op::v3::نشر>(sinks_4d, sink_نشر_shape,
                                                                   ov::op::نشر_Type::BIDIRECTIONAL);
        softmax_داخلہ = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{logits, sink_عمود}, 3);

        خودکار softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_داخلہ, -1);
        خودکار original_آخری_dim = std::make_shared<ov::op::v8::قاش>(logits_shape, three, four, one, shape_axis);
        خودکار res = std::make_shared<ov::op::v8::قاش>(softmax_with_sink, zero, original_آخری_dim, one, three);

        واپس rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    خودکار res = std::make_shared<ov::op::v8::Softmax>(softmax_داخلہ, -1);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
