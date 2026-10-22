#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml.h"

#ضمن <memory>
#ضمن <openvino/core/type.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/sigmoid.hpp>
#ضمن <openvino/op/swish.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/util/precision_sensitive_attribute.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

نطاق {

منطقي is_silu_div_نمونو(ثابت ov::Output<ov::Node> & numerator,
                         ثابت ov::Output<ov::Node> & denominator,
                         ثابت ڳنڍ_Context & context) {
    إذا (context.get_داخلا_سائيز() != 2) {
        أرجع خطأ;
    }

    ثابت خودڪار * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_داخلا_op_params(0));
    إذا (unary_op == خالي_اشارو || *unary_op != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }

    ثابت خودڪار denom_ڳنڍ = denominator.get_ڳنڍ_shared_ptr();

    إذا (خودڪار swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_ڳنڍ_shared_ptr())) {
        أرجع swish->داخلا_قدر(0).get_ڳنڍ_shared_ptr() == denom_ڳنڍ;
    }
    أرجع خطأ;
}

ov::Output<ov::Node> repeat_داخلا_منزل_ميلاپ(ثابت ڳنڍ_Context & context,
                                           ثابت ov::Output<ov::Node> & داخلا_2,
                                           ثابت ov::Output<ov::Node> & هدف,
                                           size_t داخلا_اشاريو) {
    ثابت خودڪار داخلا_shape = context.get_داخلا_shape(داخلا_اشاريو);
    ثابت خودڪار هدف_shape = context.get_داخلا_shape(0);

    إذا (داخلا_shape == هدف_shape) {
        أرجع داخلا_2;
    }

    إذا (داخلا_shape.rank().is_static() && هدف_shape.rank().is_static()) {
        ثابت خودڪار rank = static_cast<size_t>(داخلا_shape.rank().get_ڊيگهائي());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = خطأ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_پڇاڙي_OP_CONVERSION_CHECK(داخلا_shape[axis].is_static() && هدف_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t داخلا_dim = داخلا_shape[axis].get_ڊيگهائي();
            ثابت int64_t هدف_dim = هدف_shape[axis].get_ڊيگهائي();

            FRONT_پڇاڙي_OP_CONVERSION_CHECK(داخلا_dim > 0 && هدف_dim > 0 && هدف_dim % داخلا_dim == 0,
                                          "DIV input shape ", داخلا_shape, " cannot repeat to match ", هدف_shape);

            repeats[axis] = هدف_dim / داخلا_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        إذا (!needs_repeat) {
            أرجع داخلا_2;
        }

        خودڪار repeats_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        أرجع std::make_shared<ov::op::v0::Tile>(داخلا_2, repeats_ڳنڍ);
    }

    خودڪار داخلا_shape_ڳنڍ = std::make_shared<ov::op::v3::Shape_جو>(داخلا_2, ov::element::i64);
    خودڪار هدف_shape_ڳنڍ = std::make_shared<ov::op::v3::Shape_جو>(هدف, ov::element::i64);
    خودڪار repeats_ڳنڍ = std::make_shared<ov::op::v1::Divide>(هدف_shape_ڳنڍ, داخلا_shape_ڳنڍ);
    أرجع std::make_shared<ov::op::v0::Tile>(داخلا_2, repeats_ڳنڍ);
}

}  // namespace

Output_ويڪٽر translate_div(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 2, 2);

    خودڪار داخلا_0 = process_view_داخلا_new(context, 0);
    خودڪار داخلا_1 = process_view_داخلا_new(context, 1);

    إذا (is_silu_div_نمونو(داخلا_0, داخلا_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(داخلا_1);
        إذا (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    داخلا_1 = repeat_داخلا_منزل_ميلاپ(context, داخلا_1, داخلا_0, 1);

    ثابت خودڪار output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = داخلا_0.get_element_type() != ov::element::f32 ||
                                 داخلا_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    إذا (use_f32_compute) {
        داخلا_0 = std::make_shared<ov::op::v0::Convert>(داخلا_0, ov::element::f32);
        داخلا_1 = std::make_shared<ov::op::v0::Convert>(داخلا_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(داخلا_0, داخلا_1);
    إذا (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_ڳنڍ_shared_ptr()->داخلا_2(0));
        ov::mark_as_precision_sensitive(res.get_ڳنڍ_shared_ptr()->داخلا_2(1));
    }
    إذا (res.get_element_type() != output_type) {
        خودڪار output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        إذا (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->داخلا_2(0));
        }
        res = output_convert;
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
