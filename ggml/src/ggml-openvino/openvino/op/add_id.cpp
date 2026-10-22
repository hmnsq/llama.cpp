#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

ಸ್ಥಿರ ov::Output<ov::Node> reshape_add_id_ಒಳಹರಿವು_to_2d(ಧ್ರುವ ov::Output<ov::Node> & ಒಳಹರಿವು,
                                                       ಧ್ರುವ ov::PartialShape & ಒಳಹರಿವು_shape,
                                                       ಧ್ರುವ std::vector<ಪೂರ್ಣಾಙ್ಕ> & dims) {
    ಧ್ರುವ auto & actual_shape = ಒಳಹರಿವು.get_partial_shape();
    ಯದಿ (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು;
    }

    ಯದಿ (ಒಳಹರಿವು_shape.rank().is_static() && ಒಳಹರಿವು_shape.rank().get_length() == 2) {
        ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(ಒಳಹರಿವು, ov::element::i64);
    ಪ್ರತ್ಯಾಗಮ std::make_shared<ov::op::v1::Reshape>(ಒಳಹರಿವು, get_dimensions(shape, dims), ಅಸತ್ಯ);
}

OutputVector translate_add_id(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 3, 3);

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto bias = process_view_ಒಳಹರಿವು_new(context, 1);
    auto ids = process_view_ಒಳಹರಿವು_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_ಒಳಹರಿವು_to_2d(bias, context.get_ಒಳಹರಿವು_shape(1), {2, 3});
    ids = reshape_add_id_ಒಳಹರಿವು_to_2d(ids, context.get_ಒಳಹರಿವು_shape(2), {2, 3});

    ಯದಿ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(ಒಳಹರಿವು, ov::element::i64), ಅಸತ್ಯ);

    ಯದಿ (selected_bias.get_element_type() != ಒಳಹರಿವು.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, ಒಳಹರಿವು.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(ಒಳಹರಿವು, selected_bias);
    ಧ್ರುವ auto output_type = context.get_output_type();
    ಯದಿ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
