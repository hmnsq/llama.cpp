#அந்தர்பாவ "fuse_moe_router.h"

#அந்தர்பாவ <openvino/core/graph_util.hpp>
#அந்தர்பாவ <openvino/core/rt_info.hpp>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/clamp.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <openvino/op/reduce_sum.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/softmax.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>
#அந்தர்பாவ <openvino/op/tile.hpp>
#அந்தர்பாவ <openvino/op/topk.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/wrap_type.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ pass {

நாமாகாஶ {

தருக்க constant_is(மாறிலி ov::Output<ov::Node> & output, மாறிலி std::vector<int64_t> & மதிப்புகள்) {
    auto முனை = ov::as_type_ptr<ov::op::v0::Constant>(output.get_முனை_shared_ptr());
    திரும்பு முனை && முனை->get_element_type().is_integral_எண்() && முனை->cast_vector<int64_t>() == மதிப்புகள்;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using நாமாகாஶ ov::pass::pattern;
    using நாமாகாஶ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::உச்சி_K>({softmax, any_உள்ளீடு()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_உள்ளீடு(), any_உள்ளீடு(), any_உள்ளீடு(), any_உள்ளீடு()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_உள்ளீடு()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_உள்ளீடு()});
    auto data = wrap_type<v0::Squeeze>({probs, any_உள்ளீடு()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_உள்ளீடு(), any_உள்ளீடு()});
    auto ids_மொத்தம் = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_உள்ளீடு(), any_உள்ளீடு()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_மொத்தம்});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_உள்ளீடு()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_உள்ளீடு()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_உள்ளீடு()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_உள்ளீடு()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_உள்ளீடு()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    மாறிலி auto callback = [=](Matcher & m) {
        மாறிலி auto & pm = m.get_pattern_மதிப்பு_map();
        மாறிலி auto முனை = [&](மாறிலி std::shared_ptr<ov::Node> & p) { திரும்பு pm.at(p).get_முனை_shared_ptr(); };
        மாறிலி auto உள்ளீடு_is = [&](மாறிலி std::shared_ptr<ov::Node> & p, size_t i, மாறிலி std::vector<int64_t> & v) {
            திரும்பு constant_is(முனை(p)->உள்ளீடு_மதிப்பு(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(முனை(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(முனை(softmax));
        auto tk = ov::as_type_ptr<v11::உச்சி_K>(முனை(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(முனை(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(முனை(clamp));
        மாறிலி auto shape = mm->get_output_partial_shape(0);
        எனில் (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_உள்ளீடு_partial_shape(0).rank() != 4 ||
            mm->get_உள்ளீடு_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::உச்சி_K::Mode::MAX ||
            tk->get_sort_type() != v11::உச்சி_K::SortType::SORT_மதிப்புகள் || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            திரும்பு பொய்;
        }
        மாறிலி int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(முனை(ids)->get_உள்ளீடு_முனை_shared_ptr(2));
        எனில் (!end || !end->get_element_type().is_integral_எண்() || ov::shape_size(end->get_shape()) != 1) {
            திரும்பு பொய்;
        }
        மாறிலி int64_t k = end->cast_vector<int64_t>()[0];
        மாறிலி auto sorted_shape = tk->get_output_partial_shape(1);
        எனில் (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !உள்ளீடு_is(probs, 1, {1, -1, experts, 1}) ||
            !உள்ளீடு_is(data, 1, {0}) || !உள்ளீடு_is(ids_2d, 1, {0, 1}) ||
            !உள்ளீடு_is(weights_4d, 1, {0}) || !உள்ளீடு_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(முனை(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(முனை(weights))->get_special_zero() ||
            !உள்ளீடு_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(முனை(gather))->get_batch_dims() != 1 ||
            !உள்ளீடு_is(data_batch, 1, {0}) || !உள்ளீடு_is(data_batch, 2, {0}) ||
            !உள்ளீடு_is(ids_மொத்தம், 1, {1}) || !உள்ளீடு_is(ids_மொத்தம், 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(முனை(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(முனை(ids_மொத்தம்))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(முனை(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(முனை(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!உள்ளீடு_is(sum, 1, {-1}) && !உள்ளீடு_is(sum, 1, {3})) ||
            !உள்ளீடு_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(முனை(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            திரும்பு பொய்;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        எனில் (!(limit->get_min() <= 0.5 * இரட்டை(k) / experts && limit->get_max() >= 2.0)) {
            திரும்பு பொய்;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        க்கு (மாறிலி auto & உள்ளீடு : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(உள்ளீடு.get_முனை()->shared_from_this());
            எனில் (!slice || உள்ளீடு.get_index() != 0 || slice->get_உள்ளீடு_size() != 5 ||
                !constant_is(slice->உள்ளீடு_மதிப்பு(1), {0}) || !constant_is(slice->உள்ளீடு_மதிப்பு(2), {k}) ||
                !constant_is(slice->உள்ளீடு_மதிப்பு(3), {1}) ||
                (!constant_is(slice->உள்ளீடு_மதிப்பு(4), {3}) && !constant_is(slice->உள்ளீடு_மதிப்பு(4), {-1}))) {
                திரும்பு பொய்;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->உள்ளீடு_மதிப்பு(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->உள்ளீடு_மதிப்பு(1), பொய், mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::உச்சி_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::உச்சி_K::Mode::MAX, v11::உச்சி_K::SortType::SORT_மதிப்புகள், tk->get_index_element_type(), பொய்);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), உண்மை);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, முனை(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_முனை(முனை(norm), weights_out);
        க்கு (மாறிலி auto & slice : slices) {
            ov::replace_முனை(slice, ids_out);
        }
        திரும்பு உண்மை;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
