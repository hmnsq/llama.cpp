#અન્તર્ભાવ "fuse_moe_router.h"

#અન્તર્ભાવ <openvino/core/graph_util.hpp>
#અન્તર્ભાવ <openvino/core/rt_info.hpp>
#અન્તર્ભાવ <openvino/op/broadcast.hpp>
#અન્તર્ભાવ <openvino/op/clamp.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/divide.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/matmul.hpp>
#અન્તર્ભાવ <openvino/op/reduce_sum.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/softmax.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>
#અન્તર્ભાવ <openvino/op/tile.hpp>
#અન્તર્ભાવ <openvino/op/topk.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/wrap_type.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ pass {

નામાકાશ {

તાર્કિક constant_is(અચલ ov::Output<ov::Node> & output, અચલ std::vector<int64_t> & કિંમતો) {
    auto ગાંઠ = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ગાંઠ_shared_ptr());
    પાછા ગાંઠ && ગાંઠ->get_element_type().is_integral_સંખ્યા() && ગાંઠ->cast_vector<int64_t>() == કિંમતો;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using નામાકાશ ov::pass::pattern;
    using નામાકાશ ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    auto logits = wrap_type<v0::MatMul>();
    auto softmax = wrap_type<v8::Softmax>({logits});
    auto topk = wrap_type<v11::ટોચ_K>({softmax, any_આગત()});
    topk->set_output_size(2);
    auto ids = wrap_type<v8::Slice>({topk->output(1), any_આગત(), any_આગત(), any_આગત(), any_આગત()});
    auto ids_2d = wrap_type<v0::Squeeze>({ids, any_આગત()});
    auto probs = wrap_type<v1::Reshape>({softmax, any_આગત()});
    auto data = wrap_type<v0::Squeeze>({probs, any_આગત()});
    auto data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_આગત(), any_આગત()});
    auto ids_ગણતરી = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_આગત(), any_આગત()});
    auto target = wrap_type<v0::Concat>({data_batch, ids_ગણતરી});
    auto broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    auto gather = wrap_type<v8::Gather>({data, broadcast, any_આગત()});
    auto weights_4d = wrap_type<v0::Unsqueeze>({gather, any_આગત()});
    auto weights = wrap_type<v1::Reshape>({weights_4d, any_આગત()});
    auto sum = wrap_type<v1::ReduceSum>({weights, any_આગત()});
    auto clamp = wrap_type<v0::Clamp>({sum});
    auto tile = wrap_type<v0::Tile>({clamp, any_આગત()});
    auto norm = wrap_type<v1::Divide>({weights, tile});

    અચલ auto callback = [=](Matcher & m) {
        અચલ auto & pm = m.get_pattern_કિંમત_map();
        અચલ auto ગાંઠ = [&](અચલ std::shared_ptr<ov::Node> & p) { પાછા pm.at(p).get_ગાંઠ_shared_ptr(); };
        અચલ auto આગત_is = [&](અચલ std::shared_ptr<ov::Node> & p, size_t i, અચલ std::vector<int64_t> & v) {
            પાછા constant_is(ગાંઠ(p)->આગત_કિંમત(i), v);
        };
        auto mm = ov::as_type_ptr<v0::MatMul>(ગાંઠ(logits));
        auto sm = ov::as_type_ptr<v8::Softmax>(ગાંઠ(softmax));
        auto tk = ov::as_type_ptr<v11::ટોચ_K>(ગાંઠ(topk));
        auto reduce = ov::as_type_ptr<v1::ReduceSum>(ગાંઠ(sum));
        auto limit = ov::as_type_ptr<v0::Clamp>(ગાંઠ(clamp));
        અચલ auto shape = mm->get_output_partial_shape(0);
        જો (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_આગત_partial_shape(0).rank() != 4 ||
            mm->get_આગત_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::ટોચ_K::Mode::MAX ||
            tk->get_sort_type() != v11::ટોચ_K::SortType::SORT_કિંમતો || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            પાછા ખોટું;
        }
        અચલ int64_t experts = shape[3].get_length();
        auto end = ov::as_type_ptr<v0::Constant>(ગાંઠ(ids)->get_આગત_ગાંઠ_shared_ptr(2));
        જો (!end || !end->get_element_type().is_integral_સંખ્યા() || ov::shape_size(end->get_shape()) != 1) {
            પાછા ખોટું;
        }
        અચલ int64_t k = end->cast_vector<int64_t>()[0];
        અચલ auto sorted_shape = tk->get_output_partial_shape(1);
        જો (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !આગત_is(probs, 1, {1, -1, experts, 1}) ||
            !આગત_is(data, 1, {0}) || !આગત_is(ids_2d, 1, {0, 1}) ||
            !આગત_is(weights_4d, 1, {0}) || !આગત_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ગાંઠ(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ગાંઠ(weights))->get_special_zero() ||
            !આગત_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ગાંઠ(gather))->get_batch_dims() != 1 ||
            !આગત_is(data_batch, 1, {0}) || !આગત_is(data_batch, 2, {0}) ||
            !આગત_is(ids_ગણતરી, 1, {1}) || !આગત_is(ids_ગણતરી, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ગાંઠ(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ગાંઠ(ids_ગણતરી))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ગાંઠ(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(ગાંઠ(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!આગત_is(sum, 1, {-1}) && !આગત_is(sum, 1, {3})) ||
            !આગત_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ગાંઠ(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            પાછા ખોટું;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        જો (!(limit->get_min() <= 0.5 * દ્વિગુણ(k) / experts && limit->get_max() >= 2.0)) {
            પાછા ખોટું;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        માટે (અચલ auto & આગત : tk->output(1).get_target_inputs()) {
            auto slice = ov::as_type_ptr<v8::Slice>(આગત.get_ગાંઠ()->shared_from_this());
            જો (!slice || આગત.get_index() != 0 || slice->get_આગત_size() != 5 ||
                !constant_is(slice->આગત_કિંમત(1), {0}) || !constant_is(slice->આગત_કિંમત(2), {k}) ||
                !constant_is(slice->આગત_કિંમત(3), {1}) ||
                (!constant_is(slice->આગત_કિંમત(4), {3}) && !constant_is(slice->આગત_કિંમત(4), {-1}))) {
                પાછા ખોટું;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto hidden = std::make_shared<v0::Squeeze>(mm->આગત_કિંમત(0), axis0);
        auto routing = std::make_shared<v0::MatMul>(hidden, mm->આગત_કિંમત(1), ખોટું, mm->get_transpose_b());
        auto probabilities = std::make_shared<v8::Softmax>(routing, -1);
        auto selected = std::make_shared<v11::ટોચ_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::ટોચ_K::Mode::MAX, v11::ટોચ_K::SortType::SORT_કિંમતો, tk->get_index_element_type(), ખોટું);
        auto total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), સાચું);
        auto normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        auto weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        auto ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ગાંઠ(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_ગાંઠ(ગાંઠ(norm), weights_out);
        માટે (અચલ auto & slice : slices) {
            ov::replace_ગાંઠ(slice, ids_out);
        }
        પાછા સાચું;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
