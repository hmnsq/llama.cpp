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

तार्किक is_silu_div_pattern(स्थिरांक ov::Output<ov::Node> & numerator,
                         स्थिरांक ov::Output<ov::Node> & denominator,
                         स्थिरांक गाठ_Context & context) {
    जर (context.get_आदान_size() != 2) {
        परत असत्य;
    }

    स्थिरांक स्वतः * unary_op = reinterpret_cast<स्थिरांक ggml_unary_op *>(context.get_आदान_op_params(0));
    जर (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        परत असत्य;
    }

    स्थिरांक स्वतः denom_गाठ = denominator.get_गाठ_shared_ptr();

    जर (स्वतः swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_गाठ_shared_ptr())) {
        परत swish->आदान_मूल्य(0).get_गाठ_shared_ptr() == denom_गाठ;
    }
    परत असत्य;
}

ov::Output<ov::Node> repeat_आदान_to_match(स्थिरांक गाठ_Context & context,
                                           स्थिरांक ov::Output<ov::Node> & आदान,
                                           स्थिरांक ov::Output<ov::Node> & target,
                                           माप_प्रकार आदान_index) {
    स्थिरांक स्वतः आदान_shape = context.get_आदान_shape(आदान_index);
    स्थिरांक स्वतः target_shape = context.get_आदान_shape(0);

    जर (आदान_shape == target_shape) {
        परत आदान;
    }

    जर (आदान_shape.rank().is_static() && target_shape.rank().is_static()) {
        स्थिरांक स्वतः rank = static_cast<माप_प्रकार>(आदान_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        तार्किक needs_repeat = असत्य;

        साठी (माप_प्रकार axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(आदान_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            स्थिरांक int64_t आदान_dim = आदान_shape[axis].get_length();
            स्थिरांक int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(आदान_dim > 0 && target_dim > 0 && target_dim % आदान_dim == 0,
                                          "DIV input shape ", आदान_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / आदान_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        जर (!needs_repeat) {
            परत आदान;
        }

        स्वतः repeats_गाठ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        परत std::make_shared<ov::op::v0::Tile>(आदान, repeats_गाठ);
    }

    स्वतः आदान_shape_गाठ = std::make_shared<ov::op::v3::ShapeOf>(आदान, ov::element::i64);
    स्वतः target_shape_गाठ = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    स्वतः repeats_गाठ = std::make_shared<ov::op::v1::Divide>(target_shape_गाठ, आदान_shape_गाठ);
    परत std::make_shared<ov::op::v0::Tile>(आदान, repeats_गाठ);
}

}  // namespace

OutputVector translate_div(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    स्वतः आदान_0 = process_view_आदान_new(context, 0);
    स्वतः आदान_1 = process_view_आदान_new(context, 1);

    जर (is_silu_div_pattern(आदान_0, आदान_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(आदान_1);
        जर (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        परत rename_outputs_with_suffix({res}, context.get_name());
    }

    आदान_1 = repeat_आदान_to_match(context, आदान_1, आदान_0, 1);

    स्थिरांक स्वतः output_type = context.get_output_type();
    स्थिरांक तार्किक use_f32_compute = आदान_0.get_element_type() != ov::element::f32 ||
                                 आदान_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    जर (use_f32_compute) {
        आदान_0 = std::make_shared<ov::op::v0::Convert>(आदान_0, ov::element::f32);
        आदान_1 = std::make_shared<ov::op::v0::Convert>(आदान_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(आदान_0, आदान_1);
    जर (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_गाठ_shared_ptr()->आदान(0));
        ov::mark_as_precision_sensitive(res.get_गाठ_shared_ptr()->आदान(1));
    }
    जर (res.get_element_type() != output_type) {
        स्वतः output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        जर (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->आदान(0));
        }
        res = output_convert;
    }
    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
