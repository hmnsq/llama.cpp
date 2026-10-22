#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node_output.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/type.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/frontend/exception.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/broadcast.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gather.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/scatter_elements_update.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/scatter_update.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/squeeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/transpose.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_set_rows(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 3, 3);

    auto data = process_view_ନିବେଶ_new(context, 0);
    auto indices = process_view_ନିବେଶ_new(context, 1);
    auto dst = process_view_ନିବେଶ_new(context, 2);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    ଅଚଳ auto indices_shape = context.get_ନିବେଶ_shape(1);
    ଅଚଳ ତାର୍କିକ multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_length() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_length() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_length() > 1));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    ଯଦି (context.is_stateful()) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), ମିଥ୍ୟା);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, data}, concat_axis);
    } ଅନ୍ୟଥା ଯଦି (multidim_indices) {
        auto updates_shape = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);

        auto indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        auto indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, ମିଥ୍ୟା);
        auto broadcasted_indices = std::make_shared<ov::op::v3::Broadcast>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, data, axes);
    } ଅନ୍ୟଥା {
        auto row_size = context.get_ନିବେଶ_shape(2)[3].get_length();
        auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
            data,
            ov::op::v0::Constant::create(ov::element::i64, {4},
                                         {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) row_size}),
            ମିଥ୍ୟା);
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);
    }

    auto dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_ଗଣ୍ଠି_shared_ptr());
    ଯଦି (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_ନିବେଶ_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    ମିଥ୍ୟା);
    }
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
