#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <climits>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node_output.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/broadcast.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/matmul.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/transpose.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/util/op_types.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_mulmat(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 2, 2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_case = context.get_op_case();

    ov::Output<ov::Node> B = process_view_ନିବେଶ_new(context, 0);
    ov::Output<ov::Node> A = process_view_ନିବେଶ_new(context, 1);

    ଯଦି (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(B, context.get_ନିବେଶ_type(1));
    }

    auto B_shape = context.get_ନିବେଶ_shape(0).to_shape();
    auto A_shape = context.get_ନିବେଶ_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    auto Z_shape = A_batch_larger ? B_shape : A_shape;
    int64_t factor = batch_large / batch_small;
    ଯଦି (factor > 1 && batch_small > 1) {
        auto batch_large_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) Z_shape[3]});
        ଯଦି (op_case == 2) {
            new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                       {(int64_t) 0, batch_large, (int64_t) Z_shape[2], (int64_t) -1});
        }

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, ସତ୍ୟ);
    }
    ଯଦି (A_batch_larger) {
        B = Z;
    } ଅନ୍ୟଥା {
        A = Z;
    }

    ତାର୍କିକ transpose_b = ସତ୍ୟ;
    ov::Output<Node> res = std::make_shared<ov::op::v0::MatMul>(A, B, ମିଥ୍ୟା, transpose_b);

    ଅଚଳ auto output_type = context.get_output_type();
    ଯଦି (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
