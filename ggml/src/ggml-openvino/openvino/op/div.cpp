#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"
#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/core/type.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/divide.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <openvino/op/sigmoid.hpp>
#അന്തര്ഭാവ <openvino/op/swish.hpp>
#അന്തര്ഭാവ <openvino/op/tile.hpp>
#അന്തര്ഭാവ <openvino/op/util/precision_sensitive_attribute.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

നാമാകാശ {

തർക്കം is_silu_div_pattern(സ്ഥിരം ov::Output<ov::Node> & numerator,
                         സ്ഥിരം ov::Output<ov::Node> & denominator,
                         സ്ഥിരം കണ്ണി_Context & context) {
    എങ്കിൽ (context.get_നിവേശം_size() != 2) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം auto * unary_op = reinterpret_cast<സ്ഥിരം ggml_unary_op *>(context.get_നിവേശം_op_params(0));
    എങ്കിൽ (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം auto denom_കണ്ണി = denominator.get_കണ്ണി_shared_ptr();

    എങ്കിൽ (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_കണ്ണി_shared_ptr())) {
        തിരികെ swish->നിവേശം_മൂല്യം(0).get_കണ്ണി_shared_ptr() == denom_കണ്ണി;
    }
    തിരികെ അസത്യം;
}

ov::Output<ov::Node> repeat_നിവേശം_to_match(സ്ഥിരം കണ്ണി_Context & context,
                                           സ്ഥിരം ov::Output<ov::Node> & നിവേശം,
                                           സ്ഥിരം ov::Output<ov::Node> & target,
                                           size_t നിവേശം_index) {
    സ്ഥിരം auto നിവേശം_shape = context.get_നിവേശം_shape(നിവേശം_index);
    സ്ഥിരം auto target_shape = context.get_നിവേശം_shape(0);

    എങ്കിൽ (നിവേശം_shape == target_shape) {
        തിരികെ നിവേശം;
    }

    എങ്കിൽ (നിവേശം_shape.rank().is_static() && target_shape.rank().is_static()) {
        സ്ഥിരം auto rank = static_cast<size_t>(നിവേശം_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        തർക്കം needs_repeat = അസത്യം;

        വേണ്ടി (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(നിവേശം_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            സ്ഥിരം int64_t നിവേശം_dim = നിവേശം_shape[axis].get_length();
            സ്ഥിരം int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(നിവേശം_dim > 0 && target_dim > 0 && target_dim % നിവേശം_dim == 0,
                                          "DIV input shape ", നിവേശം_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / നിവേശം_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        എങ്കിൽ (!needs_repeat) {
            തിരികെ നിവേശം;
        }

        auto repeats_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        തിരികെ std::make_shared<ov::op::v0::Tile>(നിവേശം, repeats_കണ്ണി);
    }

    auto നിവേശം_shape_കണ്ണി = std::make_shared<ov::op::v3::ShapeOf>(നിവേശം, ov::element::i64);
    auto target_shape_കണ്ണി = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_കണ്ണി = std::make_shared<ov::op::v1::Divide>(target_shape_കണ്ണി, നിവേശം_shape_കണ്ണി);
    തിരികെ std::make_shared<ov::op::v0::Tile>(നിവേശം, repeats_കണ്ണി);
}

}  // namespace

OutputVector translate_div(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    auto നിവേശം_0 = process_view_നിവേശം_new(context, 0);
    auto നിവേശം_1 = process_view_നിവേശം_new(context, 1);

    എങ്കിൽ (is_silu_div_pattern(നിവേശം_0, നിവേശം_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(നിവേശം_1);
        എങ്കിൽ (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        തിരികെ rename_outputs_with_suffix({res}, context.get_name());
    }

    നിവേശം_1 = repeat_നിവേശം_to_match(context, നിവേശം_1, നിവേശം_0, 1);

    സ്ഥിരം auto output_type = context.get_output_type();
    സ്ഥിരം തർക്കം use_f32_compute = നിവേശം_0.get_element_type() != ov::element::f32 ||
                                 നിവേശം_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    എങ്കിൽ (use_f32_compute) {
        നിവേശം_0 = std::make_shared<ov::op::v0::Convert>(നിവേശം_0, ov::element::f32);
        നിവേശം_1 = std::make_shared<ov::op::v0::Convert>(നിവേശം_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(നിവേശം_0, നിവേശം_1);
    എങ്കിൽ (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_കണ്ണി_shared_ptr()->നിവേശം(0));
        ov::mark_as_precision_sensitive(res.get_കണ്ണി_shared_ptr()->നിവേശം(1));
    }
    എങ്കിൽ (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        എങ്കിൽ (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->നിവേശം(0));
        }
        res = output_convert;
    }
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
