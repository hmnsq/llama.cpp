#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/type.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/divide.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/sigmoid.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/swish.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/tile.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/util/precision_sensitive_attribute.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

ନାମାକାଶ {

ତାର୍କିକ is_silu_div_pattern(ଅଚଳ ov::Output<ov::Node> & numerator,
                         ଅଚଳ ov::Output<ov::Node> & denominator,
                         ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଯଦି (context.get_ନିବେଶ_size() != 2) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ auto * unary_op = reinterpret_cast<ଅଚଳ ggml_unary_op *>(context.get_ନିବେଶ_op_params(0));
    ଯଦି (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ auto denom_ଗଣ୍ଠି = denominator.get_ଗଣ୍ଠି_shared_ptr();

    ଯଦି (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_ଗଣ୍ଠି_shared_ptr())) {
        ଫେରନ୍ତୁ swish->ନିବେଶ_ମୂଲ୍ୟ(0).get_ଗଣ୍ଠି_shared_ptr() == denom_ଗଣ୍ଠି;
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ov::Output<ov::Node> repeat_ନିବେଶ_to_match(ଅଚଳ ଗଣ୍ଠି_Context & context,
                                           ଅଚଳ ov::Output<ov::Node> & ନିବେଶ,
                                           ଅଚଳ ov::Output<ov::Node> & target,
                                           size_t ନିବେଶ_index) {
    ଅଚଳ auto ନିବେଶ_shape = context.get_ନିବେଶ_shape(ନିବେଶ_index);
    ଅଚଳ auto target_shape = context.get_ନିବେଶ_shape(0);

    ଯଦି (ନିବେଶ_shape == target_shape) {
        ଫେରନ୍ତୁ ନିବେଶ;
    }

    ଯଦି (ନିବେଶ_shape.rank().is_static() && target_shape.rank().is_static()) {
        ଅଚଳ auto rank = static_cast<size_t>(ନିବେଶ_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        ତାର୍କିକ needs_repeat = ମିଥ୍ୟା;

        ପାଇଁ (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(ନିବେଶ_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ଅଚଳ int64_t ନିବେଶ_dim = ନିବେଶ_shape[axis].get_length();
            ଅଚଳ int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(ନିବେଶ_dim > 0 && target_dim > 0 && target_dim % ନିବେଶ_dim == 0,
                                          "DIV input shape ", ନିବେଶ_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / ନିବେଶ_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        ଯଦି (!needs_repeat) {
            ଫେରନ୍ତୁ ନିବେଶ;
        }

        auto repeats_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        ଫେରନ୍ତୁ std::make_shared<ov::op::v0::Tile>(ନିବେଶ, repeats_ଗଣ୍ଠି);
    }

    auto ନିବେଶ_shape_ଗଣ୍ଠି = std::make_shared<ov::op::v3::ShapeOf>(ନିବେଶ, ov::element::i64);
    auto target_shape_ଗଣ୍ଠି = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Divide>(target_shape_ଗଣ୍ଠି, ନିବେଶ_shape_ଗଣ୍ଠି);
    ଫେରନ୍ତୁ std::make_shared<ov::op::v0::Tile>(ନିବେଶ, repeats_ଗଣ୍ଠି);
}

}  // namespace

OutputVector translate_div(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    auto ନିବେଶ_0 = process_view_ନିବେଶ_new(context, 0);
    auto ନିବେଶ_1 = process_view_ନିବେଶ_new(context, 1);

    ଯଦି (is_silu_div_pattern(ନିବେଶ_0, ନିବେଶ_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(ନିବେଶ_1);
        ଯଦି (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
    }

    ନିବେଶ_1 = repeat_ନିବେଶ_to_match(context, ନିବେଶ_1, ନିବେଶ_0, 1);

    ଅଚଳ auto output_type = context.get_output_type();
    ଅଚଳ ତାର୍କିକ use_f32_compute = ନିବେଶ_0.get_element_type() != ov::element::f32 ||
                                 ନିବେଶ_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    ଯଦି (use_f32_compute) {
        ନିବେଶ_0 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_0, ov::element::f32);
        ନିବେଶ_1 = std::make_shared<ov::op::v0::Convert>(ନିବେଶ_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(ନିବେଶ_0, ନିବେଶ_1);
    ଯଦି (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_ଗଣ୍ଠି_shared_ptr()->ନିବେଶ(0));
        ov::mark_as_precision_sensitive(res.get_ଗଣ୍ଠି_shared_ptr()->ନିବେଶ(1));
    }
    ଯଦି (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        ଯଦି (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->ନିବେଶ(0));
        }
        res = output_convert;
    }
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
