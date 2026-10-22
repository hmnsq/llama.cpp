#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/exception.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/range.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reduce_prod.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/scatter_update.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

// GGML SET writes src1 into a view of src0 and returns the updated tensor.
Output_ਸਦਿਸ਼ translate_ਸੈੱਟ(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ਸਵੈਚਾਲਿਤ dst = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ src = process_view_ਨਿਵੇਸ਼_new(context, 1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());

    ਅਚਲ ਸਵੈਚਾਲਿਤ dst_stride = context.get_ਨਿਵੇਸ਼_stride(0);
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(dst_stride.size() >= 4, "SET requires 4D destination strides");

    ਅਚਲ ਸਵੈਚਾਲਿਤ * op_params = reinterpret_cast<ਅਚਲ uint32_t *>(context.get_output_op_params());
    ਅਚਲ size_t ਵਿੱਥ = static_cast<size_t>(op_params[3]);

    ਅਚਲ size_t elem_ਮਾਪ = dst_stride.back();
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(elem_ਮਾਪ != 0 && ਵਿੱਥ % elem_ਮਾਪ == 0,
                                  "SET offset must be aligned to destination element size");

    ਅਚਲ int64_t ਵਿੱਥ_elems = static_cast<int64_t>(ਵਿੱਥ / elem_ਮਾਪ);

    ਸਵੈਚਾਲਿਤ dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ਝੂਠ);

    ਸਵੈਚਾਲਿਤ src_flat = std::make_shared<ov::op::v1::Reshape>(
        src,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        ਝੂਠ);

    ਸਵੈਚਾਲਿਤ src_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(src_flat, ov::element::i64);
    ਸਵੈਚਾਲਿਤ src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
        ਝੂਠ);

    ਸਵੈਚਾਲਿਤ start = ov::op::v0::Constant::create(ov::element::i64, {}, {ਵਿੱਥ_elems});
    ਸਵੈਚਾਲਿਤ stop = std::make_shared<ov::op::v1::ਜੋੜੋ>(start, src_len);
    ਸਵੈਚਾਲਿਤ ਕਦਮ = ov::op::v0::Constant::create(ov::element::i64, {}, {1});

    ਸਵੈਚਾਲਿਤ indices = std::make_shared<ov::op::v4::Range>(start, stop, ਕਦਮ, ov::element::i64);
    ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    ਸਵੈਚਾਲਿਤ updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    ਸਵੈਚਾਲਿਤ dst_shape = std::make_shared<ov::op::v3::Shape_ਦਾ>(dst, ov::element::i64);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, ਝੂਠ);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
