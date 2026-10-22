#شامل "fuse_moe_router.h"

#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/rt_info.hpp>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/clamp.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/softmax.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/tile.hpp>
#شامل <openvino/op/topk.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام pass {

فضاینام {

منطقی constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & مقادیر) {
    auto گره = ov::as_type_ptr<ov::op::v0::Constant>(output.get_گره_shared_ptr());
    بازگشت گره && گره->get_element_type().is_integral_عدد() && گره->cast_vector<int64_t>() == مقادیر;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using فضاینام ov::pass::pattern;
    using فضاینام ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::بالاترین_K>({softmax, any_درون_داد()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_درون_داد(), any_درون_داد(), any_درون_داد(), any_درون_داد()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_درون_داد()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_درون_داد()});
    auto data = wrap_type<v0::Squeeze>({probs, any_درون_داد()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_درون_داد(), any_درون_داد()});
    auto ids_شمار = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_درون_داد(), any_درون_داد()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_شمار});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_درون_داد()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_درون_داد()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_درون_داد()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_درون_داد()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_درون_داد()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ثابت auto callback = [=](Matcher & m) {
        ثابت auto & pm = m.get_pattern_مقدار_map();
        ثابت auto گره = [&](ثابت std::shared_ptr<ov::Node> & p) { بازگشت pm.at(p).get_گره_shared_ptr(); };
        ثابت auto درون_داد_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            بازگشت constant_is(گره(p)->درون_داد_مقدار(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(گره(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(گره(softmax));
        auto tk = ov::as_type_ptr<v11::بالاترین_K>(گره(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(گره(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(گره(clamp));
        ثابت auto shape = mm->get_output_partial_shape(0);
        اگر (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_درون_داد_partial_shape(0).rank() != 4 ||
            mm->get_درون_داد_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::بالاترین_K::Mode::MAX ||
            tk->get_sort_type() != v11::بالاترین_K::SortType::SORT_مقادیر || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            بازگشت نادرست;
        }
        ثابت int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(گره(ids)->get_درون_داد_گره_shared_ptr(2));
        اگر (!end || !end->get_element_type().is_integral_عدد() || ov::shape_size(end->get_shape()) != 1) {
            بازگشت نادرست;
        }
        ثابت int64_t k = end->cast_vector<int64_t>()[0];
        ثابت auto sorted_shape = tk->get_output_partial_shape(1);
        اگر (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !درون_داد_is(probs, 1, {1, -1, experts, 1}) ||
            !درون_داد_is(data, 1, {0}) || !درون_داد_is(ids_2d, 1, {0, 1}) ||
            !درون_داد_is(weights_4d, 1, {0}) || !درون_داد_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(گره(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(گره(weights))->get_special_zero() ||
            !درون_داد_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(گره(gather))->get_batch_dims() != 1 ||
            !درون_داد_is(data_batch, 1, {0}) || !درون_داد_is(data_batch, 2, {0}) ||
            !درون_داد_is(ids_شمار, 1, {1}) || !درون_داد_is(ids_شمار, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(گره(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(گره(ids_شمار))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(گره(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(گره(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!درون_داد_is(sum, 1, {-1}) && !درون_داد_is(sum, 1, {3})) ||
            !درون_داد_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(گره(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            بازگشت نادرست;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        اگر (!(limit->get_min() <= 0.5 * دوگانه(k) / experts && limit->get_max() >= 2.0)) {
            بازگشت نادرست;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        برای (ثابت auto & درون_داد : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(درون_داد.get_گره()->shared_from_this());
            اگر (!slice || درون_داد.get_index() != 0 || slice->get_درون_داد_size() != 5 ||
                !constant_is(slice->درون_داد_مقدار(1), {0}) || !constant_is(slice->درون_داد_مقدار(2), {k}) ||
                !constant_is(slice->درون_داد_مقدار(3), {1}) ||
                (!constant_is(slice->درون_داد_مقدار(4), {3}) && !constant_is(slice->درون_داد_مقدار(4), {-1}))) {
                بازگشت نادرست;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->درون_داد_مقدار(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->درون_داد_مقدار(1), نادرست, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::بالاترین_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::بالاترین_K::Mode::MAX, v11::بالاترین_K::SortType::SORT_مقادیر, tk->get_index_element_type(), نادرست);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), درست);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, گره(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_گره(گره(norm), weights_out);
        برای (ثابت auto & slice : slices) {
            ov::replace_گره(slice, ids_out);
        }
        بازگشت درست;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
