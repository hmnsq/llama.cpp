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

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

نامخلا {

منطقی is_silu_div_نمونہ(ثابت ov::Output<ov::Node> & numerator,
                         ثابت ov::Output<ov::Node> & denominator,
                         ثابت گرہ_Context & context) {
    اگر (context.get_داخلہ_سائز() != 2) {
        واپس جھوٹ;
    }

    ثابت خودکار * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_داخلہ_op_params(0));
    اگر (unary_op == خالی_اشاریہ || *unary_op != GGML_UNARY_OP_SILU) {
        واپس جھوٹ;
    }

    ثابت خودکار denom_گرہ = denominator.get_گرہ_shared_ptr();

    اگر (خودکار swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_گرہ_shared_ptr())) {
        واپس swish->داخلہ_قدر(0).get_گرہ_shared_ptr() == denom_گرہ;
    }
    واپس جھوٹ;
}

ov::Output<ov::Node> repeat_داخلہ_مقصد_مطابقت(ثابت گرہ_Context & context,
                                           ثابت ov::Output<ov::Node> & داخلہ,
                                           ثابت ov::Output<ov::Node> & ہدف,
                                           size_t داخلہ_اشاریہ) {
    ثابت خودکار داخلہ_shape = context.get_داخلہ_shape(داخلہ_اشاریہ);
    ثابت خودکار ہدف_shape = context.get_داخلہ_shape(0);

    اگر (داخلہ_shape == ہدف_shape) {
        واپس داخلہ;
    }

    اگر (داخلہ_shape.rank().is_static() && ہدف_shape.rank().is_static()) {
        ثابت خودکار rank = static_cast<size_t>(داخلہ_shape.rank().get_لمبائی());
        std::vector<int64_t> repeats(rank, 1);
        منطقی needs_repeat = جھوٹ;

        برائے (size_t axis = 0; axis < rank; ++axis) {
            FRONT_انتہا_OP_CONVERSION_CHECK(داخلہ_shape[axis].is_static() && ہدف_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t داخلہ_dim = داخلہ_shape[axis].get_لمبائی();
            ثابت int64_t ہدف_dim = ہدف_shape[axis].get_لمبائی();

            FRONT_انتہا_OP_CONVERSION_CHECK(داخلہ_dim > 0 && ہدف_dim > 0 && ہدف_dim % داخلہ_dim == 0,
                                          "DIV input shape ", داخلہ_shape, " cannot repeat to match ", ہدف_shape);

            repeats[axis] = ہدف_dim / داخلہ_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        اگر (!needs_repeat) {
            واپس داخلہ;
        }

        خودکار repeats_گرہ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        واپس std::make_shared<ov::op::v0::Tile>(داخلہ, repeats_گرہ);
    }

    خودکار داخلہ_shape_گرہ = std::make_shared<ov::op::v3::Shape_کا>(داخلہ, ov::element::i64);
    خودکار ہدف_shape_گرہ = std::make_shared<ov::op::v3::Shape_کا>(ہدف, ov::element::i64);
    خودکار repeats_گرہ = std::make_shared<ov::op::v1::Divide>(ہدف_shape_گرہ, داخلہ_shape_گرہ);
    واپس std::make_shared<ov::op::v0::Tile>(داخلہ, repeats_گرہ);
}

}  // namespace

Output_سمتیہ translate_div(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    خودکار داخلہ_0 = process_view_داخلہ_new(context, 0);
    خودکار داخلہ_1 = process_view_داخلہ_new(context, 1);

    اگر (is_silu_div_نمونہ(داخلہ_0, داخلہ_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(داخلہ_1);
        اگر (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        واپس rename_outputs_with_suffix({res}, context.get_name());
    }

    داخلہ_1 = repeat_داخلہ_مقصد_مطابقت(context, داخلہ_1, داخلہ_0, 1);

    ثابت خودکار output_type = context.get_output_type();
    ثابت منطقی use_f32_compute = داخلہ_0.get_element_type() != ov::element::f32 ||
                                 داخلہ_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    اگر (use_f32_compute) {
        داخلہ_0 = std::make_shared<ov::op::v0::Convert>(داخلہ_0, ov::element::f32);
        داخلہ_1 = std::make_shared<ov::op::v0::Convert>(داخلہ_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(داخلہ_0, داخلہ_1);
    اگر (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_گرہ_shared_ptr()->داخلہ(0));
        ov::mark_as_precision_sensitive(res.get_گرہ_shared_ptr()->داخلہ(1));
    }
    اگر (res.get_element_type() != output_type) {
        خودکار output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        اگر (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->داخلہ(0));
        }
        res = output_convert;
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
