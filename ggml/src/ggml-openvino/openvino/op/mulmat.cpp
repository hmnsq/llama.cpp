#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <climits>
#شامل <cstdint>
#شامل <memory>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/op/util/op_types.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_mulmat(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 2, 2);

    صحیح op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_داخلہ_new(context, 0);
    ov::Output<ov::Node> A = process_view_داخلہ_new(context, 1);

    اگر (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_داخلہ_type(1));
    }

    خودکار B_shape = context.get_داخلہ_shape(0).مقصد_shape();
    خودکار A_shape = context.get_داخلہ_shape(1).مقصد_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    خودکار A_batch_larger = A_batch > B_batch;
    خودکار batch_large = A_batch_larger ? A_batch : B_batch;
    خودکار batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    خودکار Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t ضربی_عدد = batch_large / batch_small;
    اگر (ضربی_عدد > 1 && batch_small > 1) {
        خودکار batch_large_گرہ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        خودکار batch_small_گرہ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        خودکار ضربی_عدد_گرہ = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{ضربی_عدد});

        خودکار unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        خودکار Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        خودکار نشر_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, ضربی_عدد, (int64_t) 1, (int64_t) 1});
        خودکار new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        اگر (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        خودکار Z_broadcasted = std::make_shared<ov::op::v3::نشر>(Z_unsqueezed, نشر_shape,
                                                                     ov::op::نشر_Type::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, سچ);
    }
    اگر (A_batch_larger) {
        B = Z;
    } ورنہ {
        A = Z;
    }

    منطقی transpose_b = سچ;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, جھوٹ, transpose_b);

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
