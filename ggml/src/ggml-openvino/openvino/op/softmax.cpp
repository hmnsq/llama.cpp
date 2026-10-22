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
    أرجع dim.is_static() && dim.get_درازی() == 1;
}

ساكن منطقي same_static_dim(ثابت ov::Dimension & lhs, ثابت ov::Dimension & rhs) {
    أرجع lhs.is_static() && rhs.is_static() && lhs.get_درازی() == rhs.get_درازی();
}

ساكن منطقي is_attention_sinks_درون_داد_shape(ثابت ov::PartialShape & candidate, ثابت ov::PartialShape & logits_shape) {
    إذا (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_درازی() != 4 ||
        logits_shape.rank().get_درازی() != 4) {
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
Output_وکتور translate_soft_max(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 3);

    عائم scale = 1.0f;
    عائم max_bias = 0.0f;
    کپی_حافظه(&scale, (عائم *) context.get_output_op_params() + 0, حجم(عائم));
    کپی_حافظه(&max_bias, (عائم *) context.get_output_op_params() + 1, حجم(عائم));

    ov::Output<ov::Node> logits = context.get_درون_داد(0);
    ثابت منطقي دومی_درون_داد_is_sinks =
        context.get_درون_داد_اندازه() == 2 && is_attention_sinks_درون_داد_shape(context.get_درون_داد_shape(1), context.get_output_shape());
    ثابت منطقي has_ماسک = context.get_درون_داد_اندازه() > 1 && !دومی_درون_داد_is_sinks;
    ثابت منطقي has_sinks = دومی_درون_داد_is_sinks || context.get_درون_داد_اندازه() > 2;
    ثابت size_t sinks_درون_داد_idx = دومی_درون_داد_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    إذا (scale != 1.0f) {
        خودکار scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<عائم>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_انتها_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_ماسک),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    إذا (has_ماسک) {
        ov::Output<ov::Node> ماسک = context.get_درون_داد(1);

        // For stateful
        std::string ماسک_name = "KQ_mask_sliced";
        إذا (context.get_درون_داد_نامها()[1].find("swa") != std::string::npos) {
            ماسک_name = "KQ_mask_swa_sliced";
        }
        إذا (context.has_درون_داد(ماسک_name)) {
            ماسک = context.get_درون_داد(ماسک_name);
        }

        إذا (ماسک.get_element_type() != logits.get_element_type()) {
            ماسک = std::make_shared<ov::op::v0::Convert>(ماسک, logits.get_element_type());
        }

        إذا (max_bias > 0.0f) {
            خودکار بیرون_shape = context.get_output_shape().مقصد_shape();
            FRONT_انتها_CHECK_IMPLEMENTED(بیرون_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ثابت uint32_t n_سر = static_cast<uint32_t>(بیرون_shape[1]);
            FRONT_انتها_CHECK_IMPLEMENTED(n_سر > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ثابت uint32_t n_سر_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<عائم>(n_سر))));
            ثابت عائم m0 = std::توان(2.0f, -(max_bias) / static_cast<عائم>(n_سر_log2));
            ثابت عائم m1 = std::توان(2.0f, -(max_bias / 2.0f) / static_cast<عائم>(n_سر_log2));

            std::vector<عائم> slopes(n_سر);
            لكل (uint32_t h = 0; h < n_سر; ++h) {
                slopes[h] = h < n_سر_log2 ? std::توان(m0, static_cast<عائم>(h + 1)) :
                                              std::توان(m1, static_cast<عائم>(2 * (h - n_سر_log2) + 1));
            }

            ov::Output<ov::Node> شیب_گره =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_سر}, slopes);
            إذا (شیب_گره.get_element_type() != ماسک.get_element_type()) {
                شیب_گره = std::make_shared<ov::op::v0::Convert>(شیب_گره, ماسک.get_element_type());
            }

            خودکار شیب_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_سر), 1, 1});
            خودکار شیب_4d = std::make_shared<ov::op::v1::Reshape>(شیب_گره, شیب_shape, خطأ);
            ماسک = std::make_shared<ov::op::v1::Multiply>(ماسک, شیب_4d);
        }

        logits = std::make_shared<ov::op::v1::جمع_کن>(logits, ماسک);
    }

    ov::Output<ov::Node> softmax_درون_داد = logits;
    إذا (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_درون_داد(sinks_درون_داد_idx);
        إذا (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        خودکار sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        خودکار sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, خطأ);

        خودکار logits_shape = std::make_shared<ov::op::v3::Shape_از>(logits, ov::element::i64);
        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        خودکار four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        خودکار shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        خودکار sink_پیشوند_shape = std::make_shared<ov::op::v8::برش>(logits_shape, zero, three, one, shape_axis);
        خودکار sink_آخرین_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار sink_پخش_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_وکتور{sink_پیشوند_shape, sink_آخرین_dim}, 0);
        خودکار sink_ستونی = std::make_shared<ov::op::v3::پخش>(sinks_4d, sink_پخش_shape,
                                                                   ov::op::پخش_Type::BIDIRECTIONAL);
        softmax_درون_داد = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{logits, sink_ستونی}, 3);

        خودکار softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_درون_داد, -1);
        خودکار original_آخرین_dim = std::make_shared<ov::op::v8::برش>(logits_shape, three, four, one, shape_axis);
        خودکار res = std::make_shared<ov::op::v8::برش>(softmax_with_sink, zero, original_آخرین_dim, one, three);

        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    خودکار res = std::make_shared<ov::op::v8::Softmax>(softmax_درون_داد, -1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
