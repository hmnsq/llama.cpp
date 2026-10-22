#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/frontend/exception.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_reshape(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    எனில் (context.get_உள்ளீடு(0).get_partial_shape().is_static() &&
        context.get_உள்ளீடு_shape(0) == context.get_output_shape()) {
        திரும்பு {context.get_உள்ளீடு(0)};
    }

    மாறிலி முழு op_case = context.get_op_case();
    மாறிலி auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_முனை;
    தேர்வு (op_case) {
    நிலை 0:
        நிறுத்து;
    நிலை 1:
    நிலை 9:
        shape[1] = -1;
        எனில் (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        நிறுத்து;
    நிலை 2:
    நிலை 3:
        shape[2] = -1;
        எனில் (op_case == 3) {
            shape[3] = 1;
        }
        நிறுத்து;
    நிலை 4:
        திரும்பு {context.get_உள்ளீடு(0).get_முனை_shared_ptr()->உள்ளீடு_மதிப்பு(0)};
    நிலை 5:
    நிலை 7:
        shape = {1, 1, -1, shape[3]};
        எனில் (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        நிறுத்து;
    நிலை 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        எனில் (context.has_உள்ளீடு("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_உள்ளீடு("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_முனை =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } இல்லையெனில் {
            shape = {1, 1, -1, shape[3]};
        }
        நிறுத்து;
    நிலை 8:
        shape[0] = -1;
        நிறுத்து;
    இயல்பு:
        FRONT_END_OP_CONVERSION_CHECK(பொய், "Unsupported RESHAPE case: ", op_case);
    }
    எனில் (!new_shape_முனை) {
        new_shape_முனை = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_உள்ளீடு(0), new_shape_முனை, பொய்);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
