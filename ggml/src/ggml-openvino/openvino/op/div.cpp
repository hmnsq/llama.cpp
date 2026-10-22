#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/type.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/sigmoid.hpp>
#अन्तर्भाव <openvino/op/swish.hpp>
#अन्तर्भाव <openvino/op/tile.hpp>
#अन्तर्भाव <openvino/op/util/precision_sensitive_attribute.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

नामाकाश {

तार्किक is_silu_div_pattern(अचर ov::Output<ov::Node> & numerator,
                         अचर ov::Output<ov::Node> & denominator,
                         अचर गाँठो_Context & context) {
    यदि (context.get_आगत_size() != 2) {
        फर्काऊ असत्य;
    }

    अचर स्वतः * unary_op = reinterpret_cast<अचर ggml_unary_op *>(context.get_आगत_op_params(0));
    यदि (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        फर्काऊ असत्य;
    }

    अचर स्वतः denom_गाँठो = denominator.get_गाँठो_shared_ptr();

    यदि (स्वतः swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_गाँठो_shared_ptr())) {
        फर्काऊ swish->आगत_मान(0).get_गाँठो_shared_ptr() == denom_गाँठो;
    }
    फर्काऊ असत्य;
}

ov::Output<ov::Node> repeat_आगत_to_match(अचर गाँठो_Context & context,
                                           अचर ov::Output<ov::Node> & आगत,
                                           अचर ov::Output<ov::Node> & target,
                                           माप_प्रकार आगत_index) {
    अचर स्वतः आगत_shape = context.get_आगत_shape(आगत_index);
    अचर स्वतः target_shape = context.get_आगत_shape(0);

    यदि (आगत_shape == target_shape) {
        फर्काऊ आगत;
    }

    यदि (आगत_shape.rank().is_static() && target_shape.rank().is_static()) {
        अचर स्वतः rank = static_cast<माप_प्रकार>(आगत_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        तार्किक needs_repeat = असत्य;

        लागि (माप_प्रकार axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(आगत_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            अचर int64_t आगत_dim = आगत_shape[axis].get_length();
            अचर int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(आगत_dim > 0 && target_dim > 0 && target_dim % आगत_dim == 0,
                                          "DIV input shape ", आगत_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / आगत_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        यदि (!needs_repeat) {
            फर्काऊ आगत;
        }

        स्वतः repeats_गाँठो = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        फर्काऊ std::make_shared<ov::op::v0::Tile>(आगत, repeats_गाँठो);
    }

    स्वतः आगत_shape_गाँठो = std::make_shared<ov::op::v3::ShapeOf>(आगत, ov::element::i64);
    स्वतः target_shape_गाँठो = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    स्वतः repeats_गाँठो = std::make_shared<ov::op::v1::Divide>(target_shape_गाँठो, आगत_shape_गाँठो);
    फर्काऊ std::make_shared<ov::op::v0::Tile>(आगत, repeats_गाँठो);
}

}  // namespace

OutputVector translate_div(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 2, 2);

    स्वतः आगत_0 = process_view_आगत_new(context, 0);
    स्वतः आगत_1 = process_view_आगत_new(context, 1);

    यदि (is_silu_div_pattern(आगत_0, आगत_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(आगत_1);
        यदि (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
    }

    आगत_1 = repeat_आगत_to_match(context, आगत_1, आगत_0, 1);

    अचर स्वतः output_type = context.get_output_type();
    अचर तार्किक use_f32_compute = आगत_0.get_element_type() != ov::element::f32 ||
                                 आगत_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    यदि (use_f32_compute) {
        आगत_0 = std::make_shared<ov::op::v0::Convert>(आगत_0, ov::element::f32);
        आगत_1 = std::make_shared<ov::op::v0::Convert>(आगत_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(आगत_0, आगत_1);
    यदि (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_गाँठो_shared_ptr()->आगत(0));
        ov::mark_as_precision_sensitive(res.get_गाँठो_shared_ptr()->आगत(1));
    }
    यदि (res.get_element_type() != output_type) {
        स्वतः output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        यदि (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->आगत(0));
        }
        res = output_convert;
    }
    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
