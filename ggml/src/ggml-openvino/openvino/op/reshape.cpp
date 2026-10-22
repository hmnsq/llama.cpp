#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstdint>
#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_سمتیہ translate_reshape(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    إذا (context.get_داخلہ(0).get_partial_shape().is_static() &&
        context.get_داخلہ_shape(0) == context.get_output_shape()) {
        أرجع {context.get_داخلہ(0)};
    }

    ثابت صحيح op_case = context.get_op_case();
    ثابت خودکار output_shape = context.get_output_shape().منزل_shape();
    std::vector<int64_t> shape(output_shape.begin(), output_shape.end());
    std::shared_ptr<ov::Node> new_shape_گرہ;
    بدل (op_case) {
    حالة 0:
        اكسر;
    حالة 1:
    حالة 9:
        shape[1] = -1;
        إذا (context.is_stateful() && op_case == 1) {
            shape.erase(shape.begin());
        }
        اكسر;
    حالة 2:
    حالة 3:
        shape[2] = -1;
        إذا (op_case == 3) {
            shape[3] = 1;
        }
        اكسر;
    حالة 4:
        أرجع {context.get_داخلہ(0).get_گرہ_shared_ptr()->داخلہ_قدر(0)};
    حالة 5:
    حالة 7:
        shape = {1, 1, -1, shape[3]};
        إذا (context.is_stateful() && op_case == 5) {
            shape.erase(shape.begin());
        }
        اكسر;
    حالة 6:
        // Recurrent inputs keep the active sequence count separate from the token count.
        إذا (context.has_داخلہ("s_copy_active_slot_len")) {
            خودکار n_تھاں_سرگرم_len = context.get_داخلہ("s_copy_active_slot_len");
            خودکار emb_ناپ = ov::op::v0::Constant::create(ov::element::i64, {1}, {shape[3]});
            خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            new_shape_گرہ =
                std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{one, n_تھاں_سرگرم_len, neg_one, emb_ناپ}, 0);
        } وإلا {
            shape = {1, 1, -1, shape[3]};
        }
        اكسر;
    حالة 8:
        shape[0] = -1;
        اكسر;
    افتراضي:
        FRONT_سرا_OP_CONVERSION_CHECK(خطأ, "Unsupported RESHAPE case: ", op_case);
    }
    إذا (!new_shape_گرہ) {
        new_shape_گرہ = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
    }
    خودکار res = std::make_shared<ov::op::v1::Reshape>(context.get_داخلہ(0), new_shape_گرہ, خطأ);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
