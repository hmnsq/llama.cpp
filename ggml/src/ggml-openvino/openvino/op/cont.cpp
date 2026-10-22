
#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_cont(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    auto src_shape = context.get_ಒಳಹರಿವು_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    ಯದಿ (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        ಒಳಹರಿವು, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), ಅಸತ್ಯ);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
