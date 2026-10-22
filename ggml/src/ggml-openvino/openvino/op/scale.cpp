#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <openvino/core/except.hpp>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/equal.hpp>
#അന്തര്ഭാവ <openvino/op/gather.hpp>
#അന്തര്ഭാവ <openvino/op/greater_eq.hpp>
#അന്തര്ഭാവ <openvino/op/if.hpp>
#അന്തര്ഭാവ <openvino/op/less.hpp>
#അന്തര്ഭാവ <openvino/op/logical_or.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/range.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>
#അന്തര്ഭാവ <openvino/op/squeeze.hpp>
#അന്തര്ഭാവ <openvino/op/unsqueeze.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_scale(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    ദശാംശം scale;
    ദശാംശം bias;
    memcpy(&scale, (ദശാംശം *) context.get_output_op_params() + 0, വലുപ്പം(ദശാംശം));
    memcpy(&bias, (ദശാംശം *) context.get_output_op_params() + 1, വലുപ്പം(ദശാംശം));

    auto scale_കണ്ണി = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ദശാംശം>{scale});

    എങ്കിൽ (context.get_op_case() == 2) {
        തിരികെ {context.get_നിവേശം(0)};
    }

    എങ്കിൽ (context.get_op_case() == 1 && context.has_നിവേശം("cache_rs_reset_len")) {
        auto cache_rs_reset_idx = context.get_നിവേശം("cache_rs_reset_idx");
        auto cache_rs_reset_len = context.get_നിവേശം("cache_rs_reset_len");

        auto cache_rs = context.get_നിവേശം(0);

        auto cache_shape = std::make_shared<ov::op::v3::ShapeOf>(cache_rs, ov::element::i64);
        auto n_slots_1d = std::make_shared<ov::op::v8::Gather>(
            cache_shape, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto n_slots = std::make_shared<ov::op::v0::Squeeze>(n_slots_1d);

        auto iota = std::make_shared<ov::op::v4::Range>(
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}), n_slots,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64);

        auto idx_plus_len = std::make_shared<ov::op::v1::Add>(cache_rs_reset_idx, cache_rs_reset_len);
        auto less_than_idx = std::make_shared<ov::op::v1::Less>(iota, cache_rs_reset_idx);
        auto greater_equal_idx_plus_len = std::make_shared<ov::op::v1::GreaterEqual>(iota, idx_plus_len);
        auto keep_mask = std::make_shared<ov::op::v1::LogicalOr>(less_than_idx, greater_equal_idx_plus_len);

        auto keep_mask_f32 = std::make_shared<ov::op::v0::Convert>(keep_mask, ov::element::f32);
        auto keep_mask_reshape = std::make_shared<ov::op::v0::Unsqueeze>(
            keep_mask_f32, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        auto cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(cache_rs, keep_mask_reshape);
        തിരികെ rename_outputs_with_suffix({cleared_cache_rs}, context.get_name());
    }

    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_നിവേശം(0), scale_കണ്ണി);

    std::shared_ptr<ov::Node> res;
    എങ്കിൽ (bias != 0.0f) {
        auto bias_കണ്ണി =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<ദശാംശം>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_കണ്ണി);
    } അല്ലെങ്കിൽ {
        res = scaled;
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
