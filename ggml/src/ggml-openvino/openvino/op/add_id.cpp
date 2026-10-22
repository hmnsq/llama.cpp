#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node_output.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gather.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

ସ୍ଥିର ov::Output<ov::Node> reshape_add_id_ନିବେଶ_to_2d(ଅଚଳ ov::Output<ov::Node> & ନିବେଶ,
                                                       ଅଚଳ ov::PartialShape & ନିବେଶ_shape,
                                                       ଅଚଳ std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> & dims) {
    ଅଚଳ auto & actual_shape = ନିବେଶ.get_partial_shape();
    ଯଦି (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        ଫେରନ୍ତୁ ନିବେଶ;
    }

    ଯଦି (ନିବେଶ_shape.rank().is_static() && ନିବେଶ_shape.rank().get_length() == 2) {
        ଫେରନ୍ତୁ ନିବେଶ;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(ନିବେଶ, ov::element::i64);
    ଫେରନ୍ତୁ std::make_shared<ov::op::v1::Reshape>(ନିବେଶ, get_dimensions(shape, dims), ମିଥ୍ୟା);
}

OutputVector translate_add_id(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 3, 3);

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto bias = process_view_ନିବେଶ_new(context, 1);
    auto ids = process_view_ନିବେଶ_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_ନିବେଶ_to_2d(bias, context.get_ନିବେଶ_shape(1), {2, 3});
    ids = reshape_add_id_ନିବେଶ_to_2d(ids, context.get_ନିବେଶ_shape(2), {2, 3});

    ଯଦି (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(ନିବେଶ, ov::element::i64), ମିଥ୍ୟା);

    ଯଦି (selected_bias.get_element_type() != ନିବେଶ.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, ନିବେଶ.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(ନିବେଶ, selected_bias);
    ଅଚଳ auto output_type = context.get_output_type();
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
