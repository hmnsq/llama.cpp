#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"
#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/type.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/divide.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/sigmoid.hpp>
#અન્તર્ભાવ <openvino/op/swish.hpp>
#અન્તર્ભાવ <openvino/op/tile.hpp>
#અન્તર્ભાવ <openvino/op/util/precision_sensitive_attribute.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

નામાકાશ {

તાર્કિક is_silu_div_pattern(અચલ ov::Output<ov::Node> & numerator,
                         અચલ ov::Output<ov::Node> & denominator,
                         અચલ ગાંઠ_Context & context) {
    જો (context.get_આગત_size() != 2) {
        પાછા ખોટું;
    }

    અચલ auto * unary_op = reinterpret_cast<અચલ ggml_unary_op *>(context.get_આગત_op_params(0));
    જો (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        પાછા ખોટું;
    }

    અચલ auto denom_ગાંઠ = denominator.get_ગાંઠ_shared_ptr();

    જો (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_ગાંઠ_shared_ptr())) {
        પાછા swish->આગત_કિંમત(0).get_ગાંઠ_shared_ptr() == denom_ગાંઠ;
    }
    પાછા ખોટું;
}

ov::Output<ov::Node> repeat_આગત_to_match(અચલ ગાંઠ_Context & context,
                                           અચલ ov::Output<ov::Node> & આગત,
                                           અચલ ov::Output<ov::Node> & target,
                                           size_t આગત_index) {
    અચલ auto આગત_shape = context.get_આગત_shape(આગત_index);
    અચલ auto target_shape = context.get_આગત_shape(0);

    જો (આગત_shape == target_shape) {
        પાછા આગત;
    }

    જો (આગત_shape.rank().is_static() && target_shape.rank().is_static()) {
        અચલ auto rank = static_cast<size_t>(આગત_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        તાર્કિક needs_repeat = ખોટું;

        માટે (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(આગત_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            અચલ int64_t આગત_dim = આગત_shape[axis].get_length();
            અચલ int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(આગત_dim > 0 && target_dim > 0 && target_dim % આગત_dim == 0,
                                          "DIV input shape ", આગત_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / આગત_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        જો (!needs_repeat) {
            પાછા આગત;
        }

        auto repeats_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        પાછા std::make_shared<ov::op::v0::Tile>(આગત, repeats_ગાંઠ);
    }

    auto આગત_shape_ગાંઠ = std::make_shared<ov::op::v3::ShapeOf>(આગત, ov::element::i64);
    auto target_shape_ગાંઠ = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_ગાંઠ = std::make_shared<ov::op::v1::Divide>(target_shape_ગાંઠ, આગત_shape_ગાંઠ);
    પાછા std::make_shared<ov::op::v0::Tile>(આગત, repeats_ગાંઠ);
}

}  // namespace

OutputVector translate_div(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    auto આગત_0 = process_view_આગત_new(context, 0);
    auto આગત_1 = process_view_આગત_new(context, 1);

    જો (is_silu_div_pattern(આગત_0, આગત_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(આગત_1);
        જો (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        પાછા rename_outputs_with_suffix({res}, context.get_name());
    }

    આગત_1 = repeat_આગત_to_match(context, આગત_1, આગત_0, 1);

    અચલ auto output_type = context.get_output_type();
    અચલ તાર્કિક use_f32_compute = આગત_0.get_element_type() != ov::element::f32 ||
                                 આગત_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    જો (use_f32_compute) {
        આગત_0 = std::make_shared<ov::op::v0::Convert>(આગત_0, ov::element::f32);
        આગત_1 = std::make_shared<ov::op::v0::Convert>(આગત_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(આગત_0, આગત_1);
    જો (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_ગાંઠ_shared_ptr()->આગત(0));
        ov::mark_as_precision_sensitive(res.get_ગાંઠ_shared_ptr()->આગત(1));
    }
    જો (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        જો (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->આગત(0));
        }
        res = output_convert;
    }
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
