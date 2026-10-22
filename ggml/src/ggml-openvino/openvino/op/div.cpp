#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"
#समावेश "ggml.h"

#समावेश <memory>
#समावेश <openvino/core/type.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/divide.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/sigmoid.hpp>
#समावेश <openvino/op/swish.hpp>
#समावेश <openvino/op/tile.hpp>
#समावेश <openvino/op/util/precision_sensitive_attribute.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

bool is_silu_div_pattern(स्थिर ov::Output<ov::Node> & numerator,
                         स्थिर ov::Output<ov::Node> & denominator,
                         स्थिर गाँठ_Context & context) {
    अगर (context.get_निवेश_आकार() != 2) {
        वापस false;
    }

    स्थिर स्वतः * unary_op = reinterpret_cast<स्थिर ggml_unary_op *>(context.get_निवेश_op_params(0));
    अगर (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        वापस false;
    }

    स्थिर स्वतः denom_गाँठ = denominator.get_गाँठ_shared_ptr();

    अगर (स्वतः swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_गाँठ_shared_ptr())) {
        वापस swish->निवेश_मूल्य(0).get_गाँठ_shared_ptr() == denom_गाँठ;
    }
    वापस false;
}

ov::Output<ov::Node> repeat_निवेश_to_match(स्थिर गाँठ_Context & context,
                                           स्थिर ov::Output<ov::Node> & निवेश,
                                           स्थिर ov::Output<ov::Node> & target,
                                           माप_प्रकार निवेश_सूचकांक) {
    स्थिर स्वतः निवेश_shape = context.get_निवेश_shape(निवेश_सूचकांक);
    स्थिर स्वतः target_shape = context.get_निवेश_shape(0);

    अगर (निवेश_shape == target_shape) {
        वापस निवेश;
    }

    अगर (निवेश_shape.rank().is_static() && target_shape.rank().is_static()) {
        स्थिर स्वतः rank = static_cast<माप_प्रकार>(निवेश_shape.rank().get_लंबाई());
        std::vector<int64_t> repeats(rank, 1);
        bool needs_repeat = false;

        क्रम (माप_प्रकार axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(निवेश_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            स्थिर int64_t निवेश_dim = निवेश_shape[axis].get_लंबाई();
            स्थिर int64_t target_dim = target_shape[axis].get_लंबाई();

            FRONT_END_OP_CONVERSION_CHECK(निवेश_dim > 0 && target_dim > 0 && target_dim % निवेश_dim == 0,
                                          "DIV input shape ", निवेश_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / निवेश_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        अगर (!needs_repeat) {
            वापस निवेश;
        }

        स्वतः repeats_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        वापस std::make_shared<ov::op::v0::Tile>(निवेश, repeats_गाँठ);
    }

    स्वतः निवेश_shape_गाँठ = std::make_shared<ov::op::v3::ShapeOf>(निवेश, ov::element::i64);
    स्वतः target_shape_गाँठ = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    स्वतः repeats_गाँठ = std::make_shared<ov::op::v1::Divide>(target_shape_गाँठ, निवेश_shape_गाँठ);
    वापस std::make_shared<ov::op::v0::Tile>(निवेश, repeats_गाँठ);
}

}  // namespace

OutputVector translate_div(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 2, 2);

    स्वतः निवेश_0 = process_view_निवेश_new(context, 0);
    स्वतः निवेश_1 = process_view_निवेश_new(context, 1);

    अगर (is_silu_div_pattern(निवेश_0, निवेश_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(निवेश_1);
        अगर (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        वापस rename_outputs_with_suffix({res}, context.get_name());
    }

    निवेश_1 = repeat_निवेश_to_match(context, निवेश_1, निवेश_0, 1);

    स्थिर स्वतः output_type = context.get_output_type();
    स्थिर bool use_f32_compute = निवेश_0.get_element_type() != ov::element::f32 ||
                                 निवेश_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    अगर (use_f32_compute) {
        निवेश_0 = std::make_shared<ov::op::v0::Convert>(निवेश_0, ov::element::f32);
        निवेश_1 = std::make_shared<ov::op::v0::Convert>(निवेश_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(निवेश_0, निवेश_1);
    अगर (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_गाँठ_shared_ptr()->निवेश(0));
        ov::mark_as_precision_sensitive(res.get_गाँठ_shared_ptr()->निवेश(1));
    }
    अगर (res.get_element_type() != output_type) {
        स्वतः output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        अगर (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->निवेश(0));
        }
        res = output_convert;
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
