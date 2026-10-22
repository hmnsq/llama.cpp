#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

ساکن ov::Output<ov::Node> reshape_جوڑو_id_داخلہ_مقصد_2d(ثابت ov::Output<ov::Node> & داخلہ,
                                                       ثابت ov::PartialShape & داخلہ_shape,
                                                       ثابت std::vector<صحیح> & dims) {
    ثابت خودکار & actual_shape = داخلہ.get_partial_shape();
    اگر (actual_shape.rank().is_static() && actual_shape.rank().get_لمبائی() == 2) {
        واپس داخلہ;
    }

    اگر (داخلہ_shape.rank().is_static() && داخلہ_shape.rank().get_لمبائی() == 2) {
        واپس داخلہ;
    }

    خودکار shape = std::make_shared<ov::op::v3::Shape_کا>(داخلہ, ov::element::i64);
    واپس std::make_shared<ov::op::v1::Reshape>(داخلہ, get_dimensions(shape, dims), جھوٹ);
}

Output_سمتیہ translate_جوڑو_id(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 3, 3);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار bias = process_view_داخلہ_new(context, 1);
    خودکار ids = process_view_داخلہ_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_جوڑو_id_داخلہ_مقصد_2d(bias, context.get_داخلہ_shape(1), {2, 3});
    ids = reshape_جوڑو_id_داخلہ_مقصد_2d(ids, context.get_داخلہ_shape(2), {2, 3});

    اگر (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::Shape_کا>(داخلہ, ov::element::i64), جھوٹ);

    اگر (selected_bias.get_element_type() != داخلہ.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, داخلہ.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::جوڑو>(داخلہ, selected_bias);
    ثابت خودکار output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
