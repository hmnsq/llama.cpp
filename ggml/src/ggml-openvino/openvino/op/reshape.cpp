#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/node_output.hpp>
#অন্তর্ভাব <openvino/frontend/exception.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_reshape(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    যদি (context.get_প্রবেশ(0).get_partial_shape().is_static() &&
        context.get_প্রবেশ_shape(0) == context.get_output_shape()) {
        প্রত্যাগম {context.get_প্রবেশ(0)};
    }

    ধ্রুব পূর্ণাঙ্ক op_case = context.get_op_case();
    ধ্রুব auto output_shape = context.get_output_shape().to_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_গ্রন্থি;
    বিকল্প (op_case) {
    পক্ষ 0:
        ভঙ্গ;
    পক্ষ 1:
    পক্ষ 9:
        shape[1] = -1;
        যদি (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        ভঙ্গ;
    পক্ষ 2:
    পক্ষ 3:
        shape[2] = -1;
        যদি (op_case == 3) {
            shape[3] = 1;
        }
        ভঙ্গ;
    পক্ষ 4:
        প্রত্যাগম {context.get_প্রবেশ(0).get_গ্রন্থি_shared_ptr()->প্রবেশ_মান(0)};
    পক্ষ 5:
    পক্ষ 7:
        shape = {1, 1, -1, shape[3]};
        যদি (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        ভঙ্গ;
    পক্ষ 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        যদি (context.has_প্রবেশ("s_copy_active_slot_len")) {
            auto n_slot_active_len = context.get_প্রবেশ("s_copy_active_slot_len");
            auto emb_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_গ্রন্থি =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, n_slot_active_len, neg_one, emb_size}, 0);
        } অন্যথা {
            shape = {1, 1, -1, shape[3]};
        }
        ভঙ্গ;
    পক্ষ 8:
        shape[0] = -1;
        ভঙ্গ;
    স্বভাব:
        FRONT_END_OP_CONVERSION_CHECK(অসত্য, "Unsupported RESHAPE case: ", op_case);
    }
    যদি (!new_shape_গ্রন্থি) {
        new_shape_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_প্রবেশ(0), new_shape_গ্রন্থি, অসত্য);
    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
