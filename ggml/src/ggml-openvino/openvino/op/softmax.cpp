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
    أرجع dim.is_static() && dim.get_اوږدتیا() == 1;
}

ساكن منطقي same_static_dim(ثابت ov::Dimension & lhs, ثابت ov::Dimension & rhs) {
    أرجع lhs.is_static() && rhs.is_static() && lhs.get_اوږدتیا() == rhs.get_اوږدتیا();
}

ساكن منطقي is_attention_sinks_ننوت_shape(ثابت ov::PartialShape & candidate, ثابت ov::PartialShape & logits_shape) {
    إذا (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_اوږدتیا() != 4 ||
        logits_shape.rank().get_اوږدتیا() != 4) {
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
Output_ویکټور translate_soft_max(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 3);

    عائم scale = 1.0f;
    عائم max_bias = 0.0f;
    حافظه_نقل(&scale, (عائم *) context.get_output_op_params() + 0, حجم(عائم));
    حافظه_نقل(&max_bias, (عائم *) context.get_output_op_params() + 1, حجم(عائم));

    ov::Output<ov::Node> logits = context.get_ننوت(0);
    ثابت منطقي دوهم_ننوت_is_sinks =
        context.get_ننوت_اندازه() == 2 && is_attention_sinks_ننوت_shape(context.get_ننوت_shape(1), context.get_output_shape());
    ثابت منطقي has_ماسک = context.get_ننوت_اندازه() > 1 && !دوهم_ننوت_is_sinks;
    ثابت منطقي has_sinks = دوهم_ننوت_is_sinks || context.get_ننوت_اندازه() > 2;
    ثابت size_t sinks_ننوت_idx = دوهم_ننوت_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    إذا (scale != 1.0f) {
        خپلکاری scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<عائم>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_پای_ټکی_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_ماسک),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    إذا (has_ماسک) {
        ov::Output<ov::Node> ماسک = context.get_ننوت(1);

        // For stateful
        std::string ماسک_name = "KQ_mask_sliced";
        إذا (context.get_ننوت_نومونه()[1].find("swa") != std::string::npos) {
            ماسک_name = "KQ_mask_swa_sliced";
        }
        إذا (context.has_ننوت(ماسک_name)) {
            ماسک = context.get_ننوت(ماسک_name);
        }

        إذا (ماسک.get_element_type() != logits.get_element_type()) {
            ماسک = std::make_shared<ov::op::v0::Convert>(ماسک, logits.get_element_type());
        }

        إذا (max_bias > 0.0f) {
            خپلکاری بهر_shape = context.get_output_shape().منزل_shape();
            FRONT_پای_ټکی_CHECK_IMPLEMENTED(بهر_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ثابت uint32_t n_سر = static_cast<uint32_t>(بهر_shape[1]);
            FRONT_پای_ټکی_CHECK_IMPLEMENTED(n_سر > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ثابت uint32_t n_سر_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<عائم>(n_سر))));
            ثابت عائم m0 = std::توان(2.0f, -(max_bias) / static_cast<عائم>(n_سر_log2));
            ثابت عائم m1 = std::توان(2.0f, -(max_bias / 2.0f) / static_cast<عائم>(n_سر_log2));

            std::vector<عائم> slopes(n_سر);
            لكل (uint32_t h = 0; h < n_سر; ++h) {
                slopes[h] = h < n_سر_log2 ? std::توان(m0, static_cast<عائم>(h + 1)) :
                                              std::توان(m1, static_cast<عائم>(2 * (h - n_سر_log2) + 1));
            }

            ov::Output<ov::Node> میلان_غوټه =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_سر}, slopes);
            إذا (میلان_غوټه.get_element_type() != ماسک.get_element_type()) {
                میلان_غوټه = std::make_shared<ov::op::v0::Convert>(میلان_غوټه, ماسک.get_element_type());
            }

            خپلکاری میلان_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_سر), 1, 1});
            خپلکاری میلان_4d = std::make_shared<ov::op::v1::Reshape>(میلان_غوټه, میلان_shape, خطأ);
            ماسک = std::make_shared<ov::op::v1::Multiply>(ماسک, میلان_4d);
        }

        logits = std::make_shared<ov::op::v1::جمع_کړه>(logits, ماسک);
    }

    ov::Output<ov::Node> softmax_ننوت = logits;
    إذا (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_ننوت(sinks_ننوت_idx);
        إذا (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        خپلکاری sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        خپلکاری sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, خطأ);

        خپلکاری logits_shape = std::make_shared<ov::op::v3::Shape_د>(logits, ov::element::i64);
        خپلکاری zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خپلکاری one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        خپلکاری four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        خپلکاری shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        خپلکاری sink_مختاړی_shape = std::make_shared<ov::op::v8::ټوټه>(logits_shape, zero, three, one, shape_axis);
        خپلکاری sink_وروستی_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری sink_خپرونه_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_ویکټور{sink_مختاړی_shape, sink_وروستی_dim}, 0);
        خپلکاری sink_ستنه = std::make_shared<ov::op::v3::خپرونه>(sinks_4d, sink_خپرونه_shape,
                                                                   ov::op::خپرونه_Type::BIDIRECTIONAL);
        softmax_ننوت = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{logits, sink_ستنه}, 3);

        خپلکاری softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_ننوت, -1);
        خپلکاری original_وروستی_dim = std::make_shared<ov::op::v8::ټوټه>(logits_shape, three, four, one, shape_axis);
        خپلکاری res = std::make_shared<ov::op::v8::ټوټه>(softmax_with_sink, zero, original_وروستی_dim, one, three);

        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    خپلکاری res = std::make_shared<ov::op::v8::Softmax>(softmax_ننوت, -1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
