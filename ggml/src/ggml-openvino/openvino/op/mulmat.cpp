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

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_mulmat(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    صحیح op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_درون_داد_new(context, 0);
    ov::Output<ov::Node> A = process_view_درون_داد_new(context, 1);

    اگر (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_درون_داد_type(1));
    }

    auto B_shape = context.get_درون_داد_shape(0).to_shape();
    auto A_shape = context.get_درون_داد_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    اگر (factor > 1 && batch_small > 1) {
        auto batch_large_گره = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_گره = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_گره = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        اگر (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, درست);
    }
    اگر (A_batch_larger) {
        B = Z;
    } وگرنه {
        A = Z;
    }

    منطقی transpose_b = درست;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, نادرست, transpose_b);

    ثابت auto output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
