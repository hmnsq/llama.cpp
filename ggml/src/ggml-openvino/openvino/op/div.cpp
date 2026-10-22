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

तार्किक is_silu_div_pattern(ध्रुव ov::Output<ov::Node> & numerator,
                         ध्रुव ov::Output<ov::Node> & denominator,
                         ध्रुव गाँठ_Context & context) {
    यदि (context.get_निवेश_size() != 2) {
        प्रत्यागम असत्य;
    }

    ध्रुव स्वतः * unary_op = reinterpret_cast<ध्रुव ggml_unary_op *>(context.get_निवेश_op_params(0));
    यदि (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        प्रत्यागम असत्य;
    }

    ध्रुव स्वतः denom_गाँठ = denominator.get_गाँठ_shared_ptr();

    यदि (स्वतः swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_गाँठ_shared_ptr())) {
        प्रत्यागम swish->निवेश_मूल्य(0).get_गाँठ_shared_ptr() == denom_गाँठ;
    }
    प्रत्यागम असत्य;
}

ov::Output<ov::Node> repeat_निवेश_to_match(ध्रुव गाँठ_Context & context,
                                           ध्रुव ov::Output<ov::Node> & निवेश,
                                           ध्रुव ov::Output<ov::Node> & target,
                                           माप_प्रकार निवेश_index) {
    ध्रुव स्वतः निवेश_shape = context.get_निवेश_shape(निवेश_index);
    ध्रुव स्वतः target_shape = context.get_निवेश_shape(0);

    यदि (निवेश_shape == target_shape) {
        प्रत्यागम निवेश;
    }

    यदि (निवेश_shape.rank().is_static() && target_shape.rank().is_static()) {
        ध्रुव स्वतः rank = static_cast<माप_प्रकार>(निवेश_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        तार्किक needs_repeat = असत्य;

        प्रति (माप_प्रकार axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(निवेश_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ध्रुव int64_t निवेश_dim = निवेश_shape[axis].get_length();
            ध्रुव int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(निवेश_dim > 0 && target_dim > 0 && target_dim % निवेश_dim == 0,
                                          "DIV input shape ", निवेश_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / निवेश_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        यदि (!needs_repeat) {
            प्रत्यागम निवेश;
        }

        स्वतः repeats_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        प्रत्यागम std::make_shared<ov::op::v0::Tile>(निवेश, repeats_गाँठ);
    }

    स्वतः निवेश_shape_गाँठ = std::make_shared<ov::op::v3::ShapeOf>(निवेश, ov::element::i64);
    स्वतः target_shape_गाँठ = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    स्वतः repeats_गाँठ = std::make_shared<ov::op::v1::Divide>(target_shape_गाँठ, निवेश_shape_गाँठ);
    प्रत्यागम std::make_shared<ov::op::v0::Tile>(निवेश, repeats_गाँठ);
}

}  // namespace

OutputVector translate_div(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    स्वतः निवेश_0 = process_view_निवेश_new(context, 0);
    स्वतः निवेश_1 = process_view_निवेश_new(context, 1);

    यदि (is_silu_div_pattern(निवेश_0, निवेश_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(निवेश_1);
        यदि (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
    }

    निवेश_1 = repeat_निवेश_to_match(context, निवेश_1, निवेश_0, 1);

    ध्रुव स्वतः output_type = context.get_output_type();
    ध्रुव तार्किक use_f32_compute = निवेश_0.get_element_type() != ov::element::f32 ||
                                 निवेश_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    यदि (use_f32_compute) {
        निवेश_0 = std::make_shared<ov::op::v0::Convert>(निवेश_0, ov::element::f32);
        निवेश_1 = std::make_shared<ov::op::v0::Convert>(निवेश_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(निवेश_0, निवेश_1);
    यदि (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_गाँठ_shared_ptr()->निवेश(0));
        ov::mark_as_precision_sensitive(res.get_गाँठ_shared_ptr()->निवेश(1));
    }
    यदि (res.get_element_type() != output_type) {
        स्वतः output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        यदि (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->निवेश(0));
        }
        res = output_convert;
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
