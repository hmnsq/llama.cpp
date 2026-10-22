#अन्तर्भाव "fuse_moe_router.h"

#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/softmax.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/tile.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

नामाकाश {

तार्किक constant_is(ध्रुव ov::Output<ov::Node> & output, ध्रुव std::vector<int64_t> & मूल्यावली) {
    स्वतः गाँठ = ov::as_type_ptr<ov::op::v0::Constant>(output.get_गाँठ_shared_ptr());
    प्रत्यागम गाँठ && गाँठ->get_element_type().is_integral_संख्या() && गाँठ->cast_vector<int64_t>() == मूल्यावली;
}

}  // namespace

FuseMoeRouter::FuseMoeRouter() {
    using नामाकाश ov::pass::pattern;
    using नामाकाश ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    स्वतः logits = wrap_type<v0::MatMul>();
    स्वतः softmax = wrap_type<v8::Softmax>({logits});
    स्वतः topk = wrap_type<v11::शीर्ष_K>({softmax, any_निवेश()});
    topk->set_output_size(2);
    स्वतः ids = wrap_type<v8::Slice>({topk->output(1), any_निवेश(), any_निवेश(), any_निवेश(), any_निवेश()});
    स्वतः ids_2d = wrap_type<v0::Squeeze>({ids, any_निवेश()});
    स्वतः probs = wrap_type<v1::Reshape>({softmax, any_निवेश()});
    स्वतः data = wrap_type<v0::Squeeze>({probs, any_निवेश()});
    स्वतः data_batch = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({data}), any_निवेश(), any_निवेश()});
    स्वतः ids_गिनती = wrap_type<v8::Gather>({wrap_type<v3::ShapeOf>({ids_2d}), any_निवेश(), any_निवेश()});
    स्वतः target = wrap_type<v0::Concat>({data_batch, ids_गिनती});
    स्वतः broadcast = wrap_type<v3::Broadcast>({ids_2d, target});
    स्वतः gather = wrap_type<v8::Gather>({data, broadcast, any_निवेश()});
    स्वतः weights_4d = wrap_type<v0::Unsqueeze>({gather, any_निवेश()});
    स्वतः weights = wrap_type<v1::Reshape>({weights_4d, any_निवेश()});
    स्वतः sum = wrap_type<v1::ReduceSum>({weights, any_निवेश()});
    स्वतः clamp = wrap_type<v0::Clamp>({sum});
    स्वतः tile = wrap_type<v0::Tile>({clamp, any_निवेश()});
    स्वतः norm = wrap_type<v1::Divide>({weights, tile});

    ध्रुव स्वतः callback = [=](Matcher & m) {
        ध्रुव स्वतः & pm = m.get_pattern_मूल्य_map();
        ध्रुव स्वतः गाँठ = [&](ध्रुव std::shared_ptr<ov::Node> & p) { प्रत्यागम pm.at(p).get_गाँठ_shared_ptr(); };
        ध्रुव स्वतः निवेश_is = [&](ध्रुव std::shared_ptr<ov::Node> & p, माप_प्रकार i, ध्रुव std::vector<int64_t> & v) {
            प्रत्यागम constant_is(गाँठ(p)->निवेश_मूल्य(i), v);
        };
        स्वतः mm = ov::as_type_ptr<v0::MatMul>(गाँठ(logits));
        स्वतः sm = ov::as_type_ptr<v8::Softmax>(गाँठ(softmax));
        स्वतः tk = ov::as_type_ptr<v11::शीर्ष_K>(गाँठ(topk));
        स्वतः reduce = ov::as_type_ptr<v1::ReduceSum>(गाँठ(sum));
        स्वतः limit = ov::as_type_ptr<v0::Clamp>(गाँठ(clamp));
        ध्रुव स्वतः shape = mm->get_output_partial_shape(0);
        यदि (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_निवेश_partial_shape(0).rank() != 4 ||
            mm->get_निवेश_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_mode() != v11::शीर्ष_K::Mode::MAX ||
            tk->get_sort_type() != v11::शीर्ष_K::SortType::SORT_मूल्यावली || tk->get_stable() ||
            tk->get_index_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().empty()) {
            प्रत्यागम असत्य;
        }
        ध्रुव int64_t experts = shape[3].get_length();
        स्वतः end = ov::as_type_ptr<v0::Constant>(गाँठ(ids)->get_निवेश_गाँठ_shared_ptr(2));
        यदि (!end || !end->get_element_type().is_integral_संख्या() || ov::shape_size(end->get_shape()) != 1) {
            प्रत्यागम असत्य;
        }
        ध्रुव int64_t k = end->cast_vector<int64_t>()[0];
        ध्रुव स्वतः sorted_shape = tk->get_output_partial_shape(1);
        यदि (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_length() < k ||
            !निवेश_is(probs, 1, {1, -1, experts, 1}) ||
            !निवेश_is(data, 1, {0}) || !निवेश_is(ids_2d, 1, {0, 1}) ||
            !निवेश_is(weights_4d, 1, {0}) || !निवेश_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(गाँठ(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(गाँठ(weights))->get_special_zero() ||
            !निवेश_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(गाँठ(gather))->get_batch_dims() != 1 ||
            !निवेश_is(data_batch, 1, {0}) || !निवेश_is(data_batch, 2, {0}) ||
            !निवेश_is(ids_गिनती, 1, {1}) || !निवेश_is(ids_गिनती, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(गाँठ(data_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(गाँठ(ids_गिनती))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(गाँठ(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::Broadcast>(गाँठ(broadcast))->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!निवेश_is(sum, 1, {-1}) && !निवेश_is(sum, 1, {3})) ||
            !निवेश_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(गाँठ(norm))->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            प्रत्यागम असत्य;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        यदि (!(limit->get_min() <= 0.5 * द्विगुण(k) / experts && limit->get_max() >= 2.0)) {
            प्रत्यागम असत्य;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        प्रति (ध्रुव स्वतः & निवेश : tk->output(1).get_target_inputs()) {
            स्वतः slice = ov::as_type_ptr<v8::Slice>(निवेश.get_गाँठ()->shared_from_this());
            यदि (!slice || निवेश.get_index() != 0 || slice->get_निवेश_size() != 5 ||
                !constant_is(slice->निवेश_मूल्य(1), {0}) || !constant_is(slice->निवेश_मूल्य(2), {k}) ||
                !constant_is(slice->निवेश_मूल्य(3), {1}) ||
                (!constant_is(slice->निवेश_मूल्य(4), {3}) && !constant_is(slice->निवेश_मूल्य(4), {-1}))) {
                प्रत्यागम असत्य;
            }
            slices.push_back(slice);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        स्वतः axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        स्वतः hidden = std::make_shared<v0::Squeeze>(mm->निवेश_मूल्य(0), axis0);
        स्वतः routing = std::make_shared<v0::MatMul>(hidden, mm->निवेश_मूल्य(1), असत्य, mm->get_transpose_b());
        स्वतः probabilities = std::make_shared<v8::Softmax>(routing, -1);
        स्वतः selected = std::make_shared<v11::शीर्ष_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::शीर्ष_K::Mode::MAX, v11::शीर्ष_K::SortType::SORT_मूल्यावली, tk->get_index_element_type(), असत्य);
        स्वतः total = std::make_shared<v1::ReduceSum>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), सत्य);
        स्वतः normalized = std::make_shared<v1::Divide>(selected->output(0), total);
        स्वतः weights_out = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        स्वतः ids_out = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, गाँठ(norm)}, {hidden, routing, probabilities, selected, total, normalized, weights_out, ids_out});
        ov::replace_गाँठ(गाँठ(norm), weights_out);
        प्रति (ध्रुव स्वतः & slice : slices) {
            ov::replace_गाँठ(slice, ids_out);
        }
        प्रत्यागम सत्य;
    };
    register_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
