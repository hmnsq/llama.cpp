#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/node.hpp>
#અન્તર્ભાવ <openvino/core/node_output.hpp>
#અન્તર્ભાવ <openvino/frontend/exception.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_reshape(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    જો (context.get_આગત(0).get_partial_shape().is_static() &&
        context.get_આગત_shape(0) == context.get_output_shape()) {
        પાછા {context.get_આગત(0)};
    }

    અચલ પૂર્ણાંક op_case = context.get_op_case();
    અચલ auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_ગાંઠ;
    પસંદગી (op_case) {
    સ્થિતિ 0:
        તોડો;
    સ્થિતિ 1:
    સ્થિતિ 9:
        shape[1] = -1;
        જો (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        તોડો;
    સ્થિતિ 2:
    સ્થિતિ 3:
        shape[2] = -1;
        જો (op_case == 3) {
            shape[3] = 1;
        }
        તોડો;
    સ્થિતિ 4:
        પાછા {context.get_આગત(0).get_ગાંઠ_shared_ptr()->આગત_કિંમત(0)};
    સ્થિતિ 5:
    સ્થિતિ 7:
        shape = {1, 1, -1, shape[3]};
        જો (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        તોડો;
    સ્થિતિ 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        જો (context.has_આગત("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_આગત("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_ગાંઠ =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } નહીંતર {
            shape = {1, 1, -1, shape[3]};
        }
        તોડો;
    સ્થિતિ 8:
        shape[0] = -1;
        તોડો;
    મૂળભૂત:
        FRONT_END_OP_CONVERSION_CHECK(ખોટું, "Unsupported RESHAPE case: ", op_case);
    }
    જો (!new_shape_ગાંઠ) {
        new_shape_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_આગત(0), new_shape_ગાંઠ, ખોટું);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
