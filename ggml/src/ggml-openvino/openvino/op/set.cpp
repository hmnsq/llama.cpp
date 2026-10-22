#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <openvino/frontend/exception.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/range.hpp>
#અન્તર્ભાવ <openvino/op/reduce_prod.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/scatter_update.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
OutputVector translate_set(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 2, 2);

    auto dst = process_view_આગત_new(context, 0);
    auto src = process_view_આગત_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    અચલ auto dst_stride = context.get_આગત_stride(0);
    FRONT_END_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    અચલ auto * op_params = reinterpret_cast<અચલ uint32_t *>(context.get_output_op_params());
    અચલ size_t offset = static_cast<size_t>(op_params[3]);

    અચલ size_t elem_size = dst_stride.back();
    FRONT_END_OP_CONVERSION_CHECK(elem_size != 0 && offset % elem_size == 0,
                                  "SET offset must be aligned to destination element size");

    અચલ int64_t offset_elems = static_cast<int64_t>(offset / elem_size);

    auto dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ખોટું);

    auto src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ખોટું);

    auto src_shape = std::make_shared<ov::op::v3::ShapeOf>(src_flat, ov::element::i64);
    auto src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        ખોટું);

    auto start = ov::op::v0::Constant::create(ov::element::i64, {}, {offset_elems});
    auto stop = std::make_shared<ov::op::v1::Add>(start, src_len);
    auto step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    auto indices = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    auto updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    auto dst_shape = std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64);
    auto res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, ખોટું);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
