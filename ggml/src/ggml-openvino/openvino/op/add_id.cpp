#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

ਸਥਿਰ ov::Output<ov::Node> reshape_ਜੋੜੋ_id_ਨਿਵੇਸ਼_ਮੰਜ਼ਿਲ_2d(ਅਚਲ ov::Output<ov::Node> & ਨਿਵੇਸ਼,
                                                       ਅਚਲ ov::PartialShape & ਨਿਵੇਸ਼_shape,
                                                       ਅਚਲ std::vector<ਪੂਰਨ> & dims) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ & actual_shape = ਨਿਵੇਸ਼.get_partial_shape();
    ਜੇ (actual_shape.rank().is_static() && actual_shape.rank().get_ਲੰਮਾਈ() == 2) {
        ਵਾਪਸ ਨਿਵੇਸ਼;
    }

    ਜੇ (ਨਿਵੇਸ਼_shape.rank().is_static() && ਨਿਵੇਸ਼_shape.rank().get_ਲੰਮਾਈ() == 2) {
        ਵਾਪਸ ਨਿਵੇਸ਼;
    }

    ਸਵੈਚਾਲਿਤ shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(ਨਿਵੇਸ਼, ov::element::i64);
    ਵਾਪਸ std::make_shared<ov::op::v1::Reshape>(ਨਿਵੇਸ਼, get_dimensions(shape, dims), ਝੂਠ);
}

Output_ਸਦਿਸ਼ translate_ਜੋੜੋ_id(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 3, 3);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ bias = process_view_ਨਿਵੇਸ਼_new(context, 1);
    ਸਵੈਚਾਲਿਤ ids = process_view_ਨਿਵੇਸ਼_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_ਜੋੜੋ_id_ਨਿਵੇਸ਼_ਮੰਜ਼ਿਲ_2d(bias, context.get_ਨਿਵੇਸ਼_shape(1), {2, 3});
    ids = reshape_ਜੋੜੋ_id_ਨਿਵੇਸ਼_ਮੰਜ਼ਿਲ_2d(ids, context.get_ਨਿਵੇਸ਼_shape(2), {2, 3});

    ਜੇ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    ਸਵੈਚਾਲਿਤ gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::Shape_ਦਾ>(ਨਿਵੇਸ਼, ov::element::i64), ਝੂਠ);

    ਜੇ (selected_bias.get_element_type() != ਨਿਵੇਸ਼.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, ਨਿਵੇਸ਼.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::ਜੋੜੋ>(ਨਿਵੇਸ਼, selected_bias);
    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
