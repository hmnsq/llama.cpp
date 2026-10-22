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
    أرجع dim.is_static() && dim.get_ڊيگهائي() == 1;
}

ساكن منطقي same_static_dim(ثابت ov::Dimension & lhs, ثابت ov::Dimension & rhs) {
    أرجع lhs.is_static() && rhs.is_static() && lhs.get_ڊيگهائي() == rhs.get_ڊيگهائي();
}

ساكن منطقي is_attention_sinks_داخلا_shape(ثابت ov::PartialShape & candidate, ثابت ov::PartialShape & logits_shape) {
    إذا (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_ڊيگهائي() != 4 ||
        logits_shape.rank().get_ڊيگهائي() != 4) {
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
Output_ويڪٽر translate_soft_max(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 3);

    عائم scale = 1.0f;
    عائم max_bias = 0.0f;
    يادگيري_نقل(&scale, (عائم *) context.get_output_op_params() + 0, حجم(عائم));
    يادگيري_نقل(&max_bias, (عائم *) context.get_output_op_params() + 1, حجم(عائم));

    ov::Output<ov::Node> logits = context.get_داخلا(0);
    ثابت منطقي ٻيو_داخلا_is_sinks =
        context.get_داخلا_سائيز() == 2 && is_attention_sinks_داخلا_shape(context.get_داخلا_shape(1), context.get_output_shape());
    ثابت منطقي has_نقاب = context.get_داخلا_سائيز() > 1 && !ٻيو_داخلا_is_sinks;
    ثابت منطقي has_sinks = ٻيو_داخلا_is_sinks || context.get_داخلا_سائيز() > 2;
    ثابت size_t sinks_داخلا_idx = ٻيو_داخلا_is_sinks ? 1 : 2;

    // Apply scale first: logits = src0 * scale
    إذا (scale != 1.0f) {
        خودڪار scale_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<عائم>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_پڇاڙي_CHECK_IMPLEMENTED(!(max_bias > 0.0f && !has_نقاب),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    إذا (has_نقاب) {
        ov::Output<ov::Node> نقاب = context.get_داخلا(1);

        // For stateful
        std::string نقاب_name = "KQ_mask_sliced";
        إذا (context.get_داخلا_نالا()[1].find("swa") != std::string::npos) {
            نقاب_name = "KQ_mask_swa_sliced";
        }
        إذا (context.has_داخلا(نقاب_name)) {
            نقاب = context.get_داخلا(نقاب_name);
        }

        إذا (نقاب.get_element_type() != logits.get_element_type()) {
            نقاب = std::make_shared<ov::op::v0::Convert>(نقاب, logits.get_element_type());
        }

        إذا (max_bias > 0.0f) {
            خودڪار ٻاهريون_shape = context.get_output_shape().منزل_shape();
            FRONT_پڇاڙي_CHECK_IMPLEMENTED(ٻاهريون_shape.size() == 4, "OpenVINO softmax ALiBi path expects rank-4 tensor");

            ثابت uint32_t n_سر = static_cast<uint32_t>(ٻاهريون_shape[1]);
            FRONT_پڇاڙي_CHECK_IMPLEMENTED(n_سر > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            ثابت uint32_t n_سر_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<عائم>(n_سر))));
            ثابت عائم m0 = std::قوت(2.0f, -(max_bias) / static_cast<عائم>(n_سر_log2));
            ثابت عائم m1 = std::قوت(2.0f, -(max_bias / 2.0f) / static_cast<عائم>(n_سر_log2));

            std::vector<عائم> slopes(n_سر);
            لكل (uint32_t h = 0; h < n_سر; ++h) {
                slopes[h] = h < n_سر_log2 ? std::قوت(m0, static_cast<عائم>(h + 1)) :
                                              std::قوت(m1, static_cast<عائم>(2 * (h - n_سر_log2) + 1));
            }

            ov::Output<ov::Node> لاهي_ڳنڍ =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_سر}, slopes);
            إذا (لاهي_ڳنڍ.get_element_type() != نقاب.get_element_type()) {
                لاهي_ڳنڍ = std::make_shared<ov::op::v0::Convert>(لاهي_ڳنڍ, نقاب.get_element_type());
            }

            خودڪار لاهي_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(n_سر), 1, 1});
            خودڪار لاهي_4d = std::make_shared<ov::op::v1::Reshape>(لاهي_ڳنڍ, لاهي_shape, خطأ);
            نقاب = std::make_shared<ov::op::v1::Multiply>(نقاب, لاهي_4d);
        }

        logits = std::make_shared<ov::op::v1::جوڙيو>(logits, نقاب);
    }

    ov::Output<ov::Node> softmax_داخلا = logits;
    إذا (has_sinks) {
        ov::Output<ov::Node> sinks = context.get_داخلا(sinks_داخلا_idx);
        إذا (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        خودڪار sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
        خودڪار sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, خطأ);

        خودڪار logits_shape = std::make_shared<ov::op::v3::Shape_جو>(logits, ov::element::i64);
        خودڪار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودڪار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودڪار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        خودڪار four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        خودڪار shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        خودڪار sink_اڳياڙي_shape = std::make_shared<ov::op::v8::ٽڪرو>(logits_shape, zero, three, one, shape_axis);
        خودڪار sink_آخري_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودڪار sink_نشر_shape = std::make_shared<ov::op::v0::Concat>(
            ov::Output_ويڪٽر{sink_اڳياڙي_shape, sink_آخري_dim}, 0);
        خودڪار sink_ڪالم = std::make_shared<ov::op::v3::نشر>(sinks_4d, sink_نشر_shape,
                                                                   ov::op::نشر_Type::BIDIRECTIONAL);
        softmax_داخلا = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{logits, sink_ڪالم}, 3);

        خودڪار softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_داخلا, -1);
        خودڪار original_آخري_dim = std::make_shared<ov::op::v8::ٽڪرو>(logits_shape, three, four, one, shape_axis);
        خودڪار res = std::make_shared<ov::op::v8::ٽڪرو>(softmax_with_sink, zero, original_آخري_dim, one, three);

        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
    خودڪار res = std::make_shared<ov::op::v8::Softmax>(softmax_داخلا, -1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
