#ଅନ୍ତର୍ଭାଵ "fuse_moe_router.h"

#ଅନ୍ତର୍ଭାଵ <openvino/core/graph_util.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/rt_info.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/broadcast.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/clamp.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/divide.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gather.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/matmul.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reduce_sum.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/softmax.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/squeeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/tile.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/topk.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/wrap_type.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ pass {

ନାମାକାଶ {

ତାର୍କିକ constant_is(ଅଚଳ ov::Output<ov::Node> & output, ଅଚଳ std::vector<int64_t> & ମୂଲ୍ୟଗୁଡ଼ିକ) {
    auto ଗଣ୍ଠି = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ଗଣ୍ଠି_shared_ptr());
    ଫେରନ୍ତୁ ଗଣ୍ଠି && ଗଣ୍ଠି->get_element_type().is_integral_ସଂଖ୍ୟା() && ଗଣ୍ଠି->cast_vector<int64_t>() == ମୂଲ୍ୟଗୁଡ଼ିକ;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using ନାମାକାଶ ov::pass::pattern;
    using ନାମାକାଶ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::ଶୀର୍ଷ_K>({softmax, any_ନିବେଶ()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_ନିବେଶ(), any_ନିବେଶ(), any_ନିବେଶ(), any_ନିବେଶ()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_ନିବେଶ()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_ନିବେଶ()});
    auto data = wrap_type<v0::Squeeze>({probs, any_ନିବେଶ()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_ନିବେଶ(), any_ନିବେଶ()});
    auto ids_ଗଣନା = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_ନିବେଶ(), any_ନିବେଶ()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_ଗଣନା});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_ନିବେଶ()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_ନିବେଶ()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_ନିବେଶ()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_ନିବେଶ()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_ନିବେଶ()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    ଅଚଳ auto callback = [=](Matcher & m) {
        ଅଚଳ auto & pm = m.get_pattern_ମୂଲ୍ୟ_map();
        ଅଚଳ auto ଗଣ୍ଠି = [&](ଅଚଳ std::shared_ptr<ov::Node> & p) { ଫେରନ୍ତୁ pm.at(p).get_ଗଣ୍ଠି_shared_ptr(); };
        ଅଚଳ auto ନିବେଶ_is = [&](ଅଚଳ std::shared_ptr<ov::Node> & p, size_t i, ଅଚଳ std::vector<int64_t> & v) {
            ଫେରନ୍ତୁ constant_is(ଗଣ୍ଠି(p)->ନିବେଶ_ମୂଲ୍ୟ(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(ଗଣ୍ଠି(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(ଗଣ୍ଠି(softmax));
        auto tk = ov::as_type_ptr<v11::ଶୀର୍ଷ_K>(ଗଣ୍ଠି(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(ଗଣ୍ଠି(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(ଗଣ୍ଠି(clamp));
        ଅଚଳ auto shape = mm->get_output_partial_shape(0);
        ଯଦି (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_ନିବେଶ_partial_shape(0).rank() != 4 ||
            mm->get_ନିବେଶ_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::ଶୀର୍ଷ_K::Mode::MAX ||
            tk->get_sort_type() != v11::ଶୀର୍ଷ_K::SortType::SORT_ମୂଲ୍ୟଗୁଡ଼ିକ || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(ଗଣ୍ଠି(ids)->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(2));
        ଯଦି (!end || !end->get_element_type().is_integral_ସଂଖ୍ୟା() || ov::shape_size(end->get_shape()) != 1) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ int64_t k = end->cast_vector<int64_t>()[0];
        ଅଚଳ auto sorted_shape = tk->get_output_partial_shape(1);
        ଯଦି (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !ନିବେଶ_is(probs, 1, {1, -1, experts, 1}) ||
            !ନିବେଶ_is(data, 1, {0}) || !ନିବେଶ_is(ids_2d, 1, {0, 1}) ||
            !ନିବେଶ_is(weights_4d, 1, {0}) || !ନିବେଶ_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ଗଣ୍ଠି(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ଗଣ୍ଠି(weights))->get_special_zero() ||
            !ନିବେଶ_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ଗଣ୍ଠି(gather))->get_batch_dims() != 1 ||
            !ନିବେଶ_is(data_batch, 1, {0}) || !ନିବେଶ_is(data_batch, 2, {0}) ||
            !ନିବେଶ_is(ids_ଗଣନା, 1, {1}) || !ନିବେଶ_is(ids_ଗଣନା, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ଗଣ୍ଠି(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ଗଣ୍ଠି(ids_ଗଣନା))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ଗଣ୍ଠି(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(ଗଣ୍ଠି(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!ନିବେଶ_is(sum, 1, {-1}) && !ନିବେଶ_is(sum, 1, {3})) ||
            !ନିବେଶ_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ଗଣ୍ଠି(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        ଯଦି (!(limit->get_min() <= 0.5 * ଦ୍ୱିଗୁଣ(k) / experts && limit->get_max() >= 2.0)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        ପାଇଁ (ଅଚଳ auto & ନିବେଶ : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(ନିବେଶ.get_ଗଣ୍ଠି()->shared_from_this());
            ଯଦି (!slice || ନିବେଶ.get_index() != 0 || slice->get_ନିବେଶ_size() != 5 ||
                !constant_is(slice->ନିବେଶ_ମୂଲ୍ୟ(1), {0}) || !constant_is(slice->ନିବେଶ_ମୂଲ୍ୟ(2), {k}) ||
                !constant_is(slice->ନିବେଶ_ମୂଲ୍ୟ(3), {1}) ||
                (!constant_is(slice->ନିବେଶ_ମୂଲ୍ୟ(4), {3}) && !constant_is(slice->ନିବେଶ_ମୂଲ୍ୟ(4), {-1}))) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->ନିବେଶ_ମୂଲ୍ୟ(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->ନିବେଶ_ମୂଲ୍ୟ(1), ମିଥ୍ୟା, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::ଶୀର୍ଷ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::ଶୀର୍ଷ_K::Mode::MAX, v11::ଶୀର୍ଷ_K::SortType::SORT_ମୂଲ୍ୟଗୁଡ଼ିକ, tk->get_index_element_type(), ମିଥ୍ୟା);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ସତ୍ୟ);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ଗଣ୍ଠି(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_ଗଣ୍ଠି(ଗଣ୍ଠି(norm), weights_out);
        ପାଇଁ (ଅଚଳ auto & slice : slices) {
            ov::replace_ଗଣ୍ଠି(slice, ids_out);
        }
        ଫେରନ୍ତୁ ସତ୍ୟ;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
