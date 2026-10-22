#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <climits>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <openvino/op/util/op_types.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_mulmat(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 2, 2);

    முழு op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_உள்ளீடு_new(context, 0);
    ov::Output<ov::Node> A = process_view_உள்ளீடு_new(context, 1);

    எனில் (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_உள்ளீடு_type(1));
    }

    auto B_shape = context.get_உள்ளீடு_shape(0).to_shape();
    auto A_shape = context.get_உள்ளீடு_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    எனில் (factor > 1 && batch_small > 1) {
        auto batch_large_முனை = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_முனை = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_முனை = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        எனில் (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, உண்மை);
    }
    எனில் (A_batch_larger) {
        B = Z;
    } இல்லையெனில் {
        A = Z;
    }

    தருக்க transpose_b = உண்மை;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, பொய், transpose_b);

    மாறிலி auto output_type = context.get_output_type();
    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
