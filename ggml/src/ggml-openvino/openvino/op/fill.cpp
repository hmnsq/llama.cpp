#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    दशांश c;
    स_नकल(&c, context.get_output_op_params(), आकार(दशांश));

    स्वतः shape = context.get_आदान_shape(0).to_shape();

    स्वतः val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    स्वतः target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    स्वतः res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
