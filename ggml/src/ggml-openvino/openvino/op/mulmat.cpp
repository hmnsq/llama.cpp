#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <climits>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/node_output.hpp>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/matmul.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/transpose.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>
#অন্তর্ভাব <openvino/op/util/op_types.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_mulmat(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 2, 2);

    পূর্ণাঙ্ক op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_প্রবেশ_new(context, 0);
    ov::Output<ov::Node> A = process_view_প্রবেশ_new(context, 1);

    যদি (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_প্রবেশ_type(1));
    }

    auto B_shape = context.get_প্রবেশ_shape(0).to_shape();
    auto A_shape = context.get_প্রবেশ_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    যদি (factor > 1 && batch_small > 1) {
        auto batch_large_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        যদি (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, সত্য);
    }
    যদি (A_batch_larger) {
        B = Z;
    } অন্যথা {
        A = Z;
    }

    তার্কিক transpose_b = সত্য;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, অসত্য, transpose_b);

    ধ্রুব auto output_type = context.get_output_type();
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
