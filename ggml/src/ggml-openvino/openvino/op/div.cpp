#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "ggml.h"

#شامل <memory>
#شامل <openvino/core/type.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/sigmoid.hpp>
#شامل <openvino/op/swish.hpp>
#شامل <openvino/op/tile.hpp>
#شامل <openvino/op/util/precision_sensitive_attribute.hpp>
#شامل <vector>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

فضاینام {

منطقی is_silu_div_pattern(ثابت ov::Output<ov::Node> & numerator,
                         ثابت ov::Output<ov::Node> & denominator,
                         ثابت گره_Context & context) {
    اگر (context.get_درون_داد_size() != 2) {
        بازگشت نادرست;
    }

    ثابت auto * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_درون_داد_op_params(0));
    اگر (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        بازگشت نادرست;
    }

    ثابت auto denom_گره = denominator.get_گره_shared_ptr();

    اگر (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_گره_shared_ptr())) {
        بازگشت swish->درون_داد_مقدار(0).get_گره_shared_ptr() == denom_گره;
    }
    بازگشت نادرست;
}

ov::Output<ov::Node> repeat_درون_داد_to_match(ثابت گره_Context & context,
                                           ثابت ov::Output<ov::Node> & درون_داد,
                                           ثابت ov::Output<ov::Node> & target,
                                           size_t درون_داد_index) {
    ثابت auto درون_داد_shape = context.get_درون_داد_shape(درون_داد_index);
    ثابت auto target_shape = context.get_درون_داد_shape(0);

    اگر (درون_داد_shape == target_shape) {
        بازگشت درون_داد;
    }

    اگر (درون_داد_shape.rank().is_static() && target_shape.rank().is_static()) {
        ثابت auto rank = static_cast<size_t>(درون_داد_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        منطقی needs_repeat = نادرست;

        برای (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(درون_داد_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t درون_داد_dim = درون_داد_shape[axis].get_length();
            ثابت int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(درون_داد_dim > 0 && target_dim > 0 && target_dim % درون_داد_dim == 0,
                                          "DIV input shape ", درون_داد_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / درون_داد_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        اگر (!needs_repeat) {
            بازگشت درون_داد;
        }

        auto repeats_گره = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        بازگشت std::make_shared<ov::op::v0::Tile>(درون_داد, repeats_گره);
    }

    auto درون_داد_shape_گره = std::make_shared<ov::op::v3::ShapeOf>(درون_داد, ov::element::i64);
    auto target_shape_گره = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_گره = std::make_shared<ov::op::v1::Divide>(target_shape_گره, درون_داد_shape_گره);
    بازگشت std::make_shared<ov::op::v0::Tile>(درون_داد, repeats_گره);
}

}  // namespace

OutputVector translate_div(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    auto درون_داد_0 = process_view_درون_داد_new(context, 0);
    auto درون_داد_1 = process_view_درون_داد_new(context, 1);

    اگر (is_silu_div_pattern(درون_داد_0, درون_داد_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(درون_داد_1);
        اگر (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        بازگشت rename_outputs_with_suffix({res}, context.get_name());
    }

    درون_داد_1 = repeat_درون_داد_to_match(context, درون_داد_1, درون_داد_0, 1);

    ثابت auto output_type = context.get_output_type();
    ثابت منطقی use_f32_compute = درون_داد_0.get_element_type() != ov::element::f32 ||
                                 درون_داد_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    اگر (use_f32_compute) {
        درون_داد_0 = std::make_shared<ov::op::v0::Convert>(درون_داد_0, ov::element::f32);
        درون_داد_1 = std::make_shared<ov::op::v0::Convert>(درون_داد_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(درون_داد_0, درون_داد_1);
    اگر (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_گره_shared_ptr()->درون_داد(0));
        ov::mark_as_precision_sensitive(res.get_گره_shared_ptr()->درون_داد(1));
    }
    اگر (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        اگر (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->درون_داد(0));
        }
        res = output_convert;
    }
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
