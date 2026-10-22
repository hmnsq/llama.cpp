#अन्तर्भाव "fuse_moe_compressed.h"

#अन्तर्भाव <cstring>
#अन्तर्भाव <limits>
#अन्तर्भाव <set>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gelu.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/swish.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/constant_folding.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/optional.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

#अन्तर्भाव "../op/gather_matmul.hpp"
#अन्तर्भाव "../op/moe_compressed.hpp"

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

नामाकाश {

संरचना dequant_inputs {
    ov::Output<ov::Node> weight;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    तार्किक has_zp = असत्य;
    तार्किक ok = असत्य;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(ध्रुव ov::Output<ov::Node> & b) {
    dequant_inputs res;

    स्वचालित ग्रन्थिः = b.get_ग्रन्थिः_shared_ptr();
    यावत् (ov::is_type<ov::op::v0::Convert>(ग्रन्थिः) || ov::is_type<ov::op::v1::Reshape>(ग्रन्थिः)) {
        ग्रन्थिः = ग्रन्थिः->get_निवेशः_ग्रन्थिः_shared_ptr(0);
    }

    स्वचालित mul = ov::as_type_ptr<ov::op::v1::Multiply>(ग्रन्थिः);
    यदि (!mul) {
        प्रत्यागम res;
    }
    res.scale = mul->निवेशः_मानम्(1);

    स्वचालित lhs = mul->get_निवेशः_ग्रन्थिः_shared_ptr(0);
    यदि (स्वचालित sub = ov::as_type_ptr<ov::op::v1::व्यवकलय>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        स्वचालित zp_ग्रन्थिः = sub->get_निवेशः_ग्रन्थिः_shared_ptr(1);
        यावत् (ov::is_type<ov::op::v0::Convert>(zp_ग्रन्थिः)) {
            zp_ग्रन्थिः = zp_ग्रन्थिः->get_निवेशः_ग्रन्थिः_shared_ptr(0);
        }
        res.zp = zp_ग्रन्थिः->output(0);
        res.has_zp = सत्य;
        lhs = sub->get_निवेशः_ग्रन्थिः_shared_ptr(0);
    }
    यावत् (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_निवेशः_ग्रन्थिः_shared_ptr(0);
    }
    यदि (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        प्रत्यागम res;
    }

    res.weight = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.weight.get_partial_shape().is_static();
    प्रत्यागम res;
}

size_t logical_k(ध्रुव ov::Shape & shape) {
    प्रत्यागम shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> खण्डिका_experts_dim1(ध्रुव ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    स्वचालित constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_ग्रन्थिः_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    ध्रुव स्वचालित & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    ध्रुव स्वचालित & type = constant->get_element_type();

    size_t row_elems = 1;
    प्रति (size_t i = 2; i < shape.size(); ++i) {
        row_elems *= shape[i];
    }
    ध्रुव size_t row_द्वयङ्काः = row_elems * type.bitwidth();
    OPENVINO_ASSERT(row_द्वयङ्काः % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    ध्रुव size_t row_अष्टकानि = row_द्वयङ्काः / 8;

    ov::Shape बहिर्गमनम्_shape = shape;
    बहिर्गमनम्_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> बहिर्गमनम्_दत्त(shape[0] * static_cast<size_t>(end - begin) * row_अष्टकानि);
    ध्रुव स्वचालित * src = static_cast<ध्रुव uint8_t *>(constant->get_दत्त_ptr());
    ध्रुव size_t src_row_अष्टकानि = shape[1] * row_अष्टकानि;
    प्रति (size_t e = 0; e < shape[0]; ++e) {
        std::स्मृतिं_प्रतिलिख(बहिर्गमनम्_दत्त.दत्त() + e * static_cast<size_t>(end - begin) * row_अष्टकानि,
                   src + e * src_row_अष्टकानि + static_cast<size_t>(begin) * row_अष्टकानि,
                   static_cast<size_t>(end - begin) * row_अष्टकानि);
    }

    प्रत्यागम std::make_shared<ov::op::v0::Constant>(type, बहिर्गमनम्_shape, बहिर्गमनम्_दत्त.दत्त());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    उपयुज्य नामाकाश ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    स्वचालित hidden_gate_m = any_निवेशः();
    स्वचालित a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_निवेशः() });
    स्वचालित a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_निवेशः() });
    स्वचालित hidden_ऊर्ध्वम्_m = any_निवेशः();
    स्वचालित a_ऊर्ध्वम्_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_ऊर्ध्वम्_m, any_निवेशः() }) }),
          any_निवेशः() });

    स्वचालित gate_w_m = any_निवेशः();
    स्वचालित ऊर्ध्वम्_w_m = any_निवेशः();
    स्वचालित अधस्तात्_w_m = any_निवेशः();
    स्वचालित ids_gate_m = any_निवेशः();
    स्वचालित ids_ऊर्ध्वम्_m = any_निवेशः();
    स्वचालित ids_अधस्तात्_m = any_निवेशः();

    स्वचालित bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_निवेशः() });
    स्वचालित gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_निवेशः() }), any_निवेशः() }) });

    स्वचालित silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    स्वचालित bgm_ऊर्ध्वम्_m = wrap_type<ov::op::internal::GatherMatmul>({ a_ऊर्ध्वम्_m, ऊर्ध्वम्_w_m, ids_ऊर्ध्वम्_m, any_निवेशः() });
    स्वचालित ऊर्ध्वम्_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_ऊर्ध्वम्_m, any_निवेशः() }), any_निवेशः() }) });
    स्वचालित swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, ऊर्ध्वम्_u_m });

    स्वचालित d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_निवेशः() }) }),
          any_निवेशः() });
    स्वचालित bgm_अधस्तात्_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, अधस्तात्_w_m, ids_अधस्तात्_m, any_निवेशः() });
    स्वचालित अधस्तात्_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_अधस्तात्_m, any_निवेशः() }), any_निवेशः() }) });

    स्वचालित routing_m = any_निवेशः();
    स्वचालित weighted_m = wrap_type<ov::op::v1::Multiply>({ अधस्तात्_u_m, routing_m });
    स्वचालित मूलम्_m = wrap_type<ov::op::v1::Reduce_योग>({ weighted_m, any_निवेशः() });

    ध्रुव स्वचालित callback = [=](Matcher & m) {
        स्वचालित & pm = m.get_आदर्शः_मानम्_map();

        ध्रुव स्वचालित gate = unwrap_dequant(pm.at(gate_w_m));
        ध्रुव स्वचालित ऊर्ध्वम् = unwrap_dequant(pm.at(ऊर्ध्वम्_w_m));
        ध्रुव स्वचालित अधस्तात् = unwrap_dequant(pm.at(अधस्तात्_w_m));
        यदि (!gate.ok || !ऊर्ध्वम्.ok || !अधस्तात्.ok) {
            प्रत्यागम असत्य;
        }

        ध्रुव स्वचालित gate_shape = gate.weight.get_shape();
        ध्रुव स्वचालित ऊर्ध्वम्_shape = ऊर्ध्वम्.weight.get_shape();
        ध्रुव स्वचालित अधस्तात्_shape = अधस्तात्.weight.get_shape();
        यदि (gate_shape != ऊर्ध्वम्_shape || gate_shape.size() < 3 || अधस्तात्_shape.size() < 3) {
            प्रत्यागम असत्य;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        यदि (gate.has_zp != अधस्तात्.has_zp || gate_shape.size() != अधस्तात्_shape.size()) {
            प्रत्यागम असत्य;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        यदि (gate.has_zp) {
            स्थिर ध्रुव std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            यदि (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(अधस्तात्.zp.get_element_type()) == 0) {
                प्रत्यागम असत्य;
            }
        }

        // Config holds a single group_size for all three projections.
        ध्रुव स्वचालित गणः_सम्बन्धि = [](ध्रुव dequant_inputs & w) {
            ध्रुव स्वचालित s = w.weight.get_shape();
            प्रत्यागम s.size() == 4 ? s[3] : logical_k(s);
        };
        यदि (गणः_सम्बन्धि(gate) != गणः_सम्बन्धि(ऊर्ध्वम्) || गणः_सम्बन्धि(gate) != गणः_सम्बन्धि(अधस्तात्)) {
            प्रत्यागम असत्य;
        }

        // all three branches must route the same hidden state through the same experts
        यदि (pm.at(hidden_gate_m) != pm.at(hidden_ऊर्ध्वम्_m)) {
            प्रत्यागम असत्य;
        }

        स्वचालित ids = pm.at(ids_अधस्तात्_m);
        ध्रुव स्वचालित ids_pshape = ids.get_partial_shape();
        यदि (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_दीर्घता() - 1].is_dynamic()) {
            प्रत्यागम असत्य;
        }
        ध्रुव size_t शिखरम्_k = ids_pshape[ids_pshape.rank().get_दीर्घता() - 1].get_दीर्घता();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        स्वचालित routing = pm.at(routing_m);
        ध्रुव स्वचालित routing_pshape = routing.get_partial_shape();
        यदि (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_दीर्घता() != 4 ||
            routing_pshape[3] != 1) {
            प्रत्यागम असत्य;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        यदि (ids_pshape.rank().get_दीर्घता() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        यदि (routing.get_partial_shape() != ids.get_partial_shape()) {
            प्रत्यागम असत्य;
        }

        ध्रुव size_t अधस्तात्_k = logical_k(अधस्तात्_shape);
        ध्रुव स्वचालित अधस्तात्_scale_shape = अधस्तात्.scale.get_shape();
        ध्रुव size_t अधस्तात्_groups = अधस्तात्_scale_shape.size() >= 3 ? अधस्तात्_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_आकार = logical_k(gate_shape);
        config.inter_आकार = gate_shape[1];
        config.अङ्कः_expert = gate_shape[0];
        config.अङ्कः_shared_expert = 0;
        config.शिखरम्_k = शिखरम्_k;
        config.गणः_आकार = अधस्तात्_groups <= 1 ? std::numeric_limits<size_t>::max() : अधस्तात्_k / अधस्तात्_groups;
        config.has_batch_dim = सत्य;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.बहिर्गमनम्_type = ov::element::dynamic;

        स्वचालित absent_zp = [] {
            स्वचालित zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            प्रत्यागम zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        ध्रुव स्वचालित a_transpose = pm.at(a_gate_m).get_ग्रन्थिः_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->निवेशः_मानम्(1));

        ध्रुव ov::Output_सदिशम् तर्काः_2 = {
            hidden, routing, ids,
            gate.weight,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            ऊर्ध्वम्.weight,    ऊर्ध्वम्.scale,     ऊर्ध्वम्.has_zp ? ऊर्ध्वम्.zp : absent_zp(),
            अधस्तात्.weight,  अधस्तात्.scale,   अधस्तात्.has_zp ? अधस्तात्.zp : absent_zp(),
        };

        स्वचालित moe = std::make_shared<ov::op::internal::MOECompressed>(तर्काः_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> परिणाम = moe->output(0);
        ध्रुव स्वचालित मूलम्_type = m.get_मेलनम्_मूलम्()->get_output_element_type(0);
        यदि (परिणाम.get_element_type() != मूलम्_type) {
            परिणाम = std::make_shared<ov::op::v0::Convert>(परिणाम, मूलम्_type);
        }

        परिणाम.get_ग्रन्थिः_shared_ptr()->समुच्चयः_friendly_name(m.get_मेलनम्_मूलम्()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), परिणाम.get_ग्रन्थिः_shared_ptr());
        ov::replace_ग्रन्थिः(m.get_मेलनम्_मूलम्(), परिणाम.get_ग्रन्थिः_shared_ptr());
        पञ्जिकाकोशः_new_ग्रन्थिः(moe);
        प्रत्यागम सत्य;
    };

    पञ्जिकाकोशः_matcher(std::make_shared<Matcher>(मूलम्_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_ऊर्ध्वम्::Fuse_Moe_Compressed_Fused_Gate_ऊर्ध्वम्() {
    उपयुज्य नामाकाश ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    स्वचालित hidden_m = any_निवेशः();
    स्वचालित a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_निवेशः() });
    स्वचालित a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_निवेशः() });

    स्वचालित gate_ऊर्ध्वम्_w_m = any_निवेशः();
    स्वचालित ids_gate_ऊर्ध्वम्_m = any_निवेशः();
    स्वचालित bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_ऊर्ध्वम्_w_m, ids_gate_ऊर्ध्वम्_m, any_निवेशः() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    स्वचालित gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_निवेशः() });
    स्वचालित gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_निवेशः() }) });

    स्वचालित gate_खण्डिका_m = wrap_type<ov::op::v8::खण्डिका>({ gu_u_m, any_निवेशः(), any_निवेशः(), any_निवेशः(), any_निवेशः() });
    स्वचालित ऊर्ध्वम्_खण्डिका_m = wrap_type<ov::op::v8::खण्डिका>({ gu_u_m, any_निवेशः(), any_निवेशः(), any_निवेशः(), any_निवेशः() });
    स्वचालित gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_खण्डिका_m });
    स्वचालित geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, ऊर्ध्वम्_खण्डिका_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    स्वचालित d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_निवेशः() }), any_निवेशः() }) }),
          any_निवेशः() });
    स्वचालित अधस्तात्_w_m = any_निवेशः();
    स्वचालित ids_अधस्तात्_m = any_निवेशः();
    स्वचालित bgm_अधस्तात्_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, अधस्तात्_w_m, ids_अधस्तात्_m, any_निवेशः() });
    // Same two shapes as the gate/up branch above.
    स्वचालित अधस्तात्_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_अधस्तात्_m }), any_निवेशः() });
    स्वचालित अधस्तात्_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ अधस्तात्_t_m, any_निवेशः() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    स्वचालित अधस्तात्_scale_m = any_निवेशः();
    स्वचालित अधस्तात्_scaled_m = wrap_type<ov::op::v1::Multiply>({ अधस्तात्_u_m, अधस्तात्_scale_m });

    स्वचालित routing_m = any_निवेशः();
    स्वचालित weighted_m = wrap_type<ov::op::v1::Multiply>({ अधस्तात्_scaled_m, routing_m });
    स्वचालित मूलम्_m = wrap_type<ov::op::v1::Reduce_योग>({ weighted_m, any_निवेशः() });

    ध्रुव स्वचालित callback = [=](Matcher & m) {
        स्वचालित & pm = m.get_आदर्शः_मानम्_map();

        ध्रुव स्वचालित gate_ऊर्ध्वम् = unwrap_dequant(pm.at(gate_ऊर्ध्वम्_w_m));
        ध्रुव स्वचालित अधस्तात् = unwrap_dequant(pm.at(अधस्तात्_w_m));
        यदि (!gate_ऊर्ध्वम्.ok || !अधस्तात्.ok) {
            प्रत्यागम असत्य;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        यदि (!gate_ऊर्ध्वम्.has_zp || !अधस्तात्.has_zp) {
            प्रत्यागम असत्य;
        }

        ध्रुव स्वचालित gate_ऊर्ध्वम्_shape = gate_ऊर्ध्वम्.weight.get_shape();
        ध्रुव स्वचालित अधस्तात्_shape = अधस्तात्.weight.get_shape();
        यदि (gate_ऊर्ध्वम्_shape.size() < 3 || अधस्तात्_shape.size() < 3 || gate_ऊर्ध्वम्_shape.size() != अधस्तात्_shape.size()) {
            प्रत्यागम असत्य;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        स्थिर ध्रुव std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        यदि (int_types.count(gate_ऊर्ध्वम्.weight.get_element_type()) == 0 ||
            int_types.count(अधस्तात्.weight.get_element_type()) == 0 ||
            int_types.count(gate_ऊर्ध्वम्.zp.get_element_type()) == 0 ||
            int_types.count(अधस्तात्.zp.get_element_type()) == 0) {
            प्रत्यागम असत्य;
        }

        ध्रुव स्वचालित गणः_सम्बन्धि = [](ध्रुव dequant_inputs & w) {
            ध्रुव स्वचालित s = w.weight.get_shape();
            प्रत्यागम s.size() == 4 ? s[3] : logical_k(s);
        };
        यदि (गणः_सम्बन्धि(gate_ऊर्ध्वम्) != गणः_सम्बन्धि(अधस्तात्)) {
            प्रत्यागम असत्य;
        }

        स्वचालित gelu_ग्रन्थिः = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_ग्रन्थिः_shared_ptr());
        यदि (!gelu_ग्रन्थिः || gelu_ग्रन्थिः->get_approximation_बहुलकम्() != ov::op::Gelu_Approximation_बहुलकम्::ERF) {
            प्रत्यागम असत्य;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        स्वचालित gate_खण्डिका = ov::as_type_ptr<ov::op::v8::खण्डिका>(pm.at(gate_खण्डिका_m).get_ग्रन्थिः_shared_ptr());
        स्वचालित ऊर्ध्वम्_खण्डिका = ov::as_type_ptr<ov::op::v8::खण्डिका>(pm.at(ऊर्ध्वम्_खण्डिका_m).get_ग्रन्थिः_shared_ptr());
        स्वचालित gate_अन्तः_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_खण्डिका->निवेशः_मानम्(2).get_ग्रन्थिः_shared_ptr());
        स्वचालित ऊर्ध्वम्_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(ऊर्ध्वम्_खण्डिका->निवेशः_मानम्(1).get_ग्रन्थिः_shared_ptr());
        यदि (!gate_अन्तः_c || !ऊर्ध्वम्_begin_c) {
            प्रत्यागम असत्य;
        }
        ध्रुव स्वचालित gate_अन्तः = gate_अन्तः_c->cast_सदिशम्<int64_t>().at(0);
        ध्रुव स्वचालित ऊर्ध्वम्_begin = ऊर्ध्वम्_begin_c->cast_सदिशम्<int64_t>().at(0);
        यदि (gate_अन्तः != ऊर्ध्वम्_begin || gate_अन्तः <= 0 || gate_अन्तः >= static_cast<int64_t>(gate_ऊर्ध्वम्_shape[1])) {
            प्रत्यागम असत्य;
        }
        ध्रुव int64_t mid = gate_अन्तः;
        ध्रुव int64_t पूर्णम् = static_cast<int64_t>(gate_ऊर्ध्वम्_shape[1]);

        स्वचालित ids = pm.at(ids_अधस्तात्_m);
        ध्रुव स्वचालित ids_pshape = ids.get_partial_shape();
        यदि (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_दीर्घता() - 1].is_dynamic()) {
            प्रत्यागम असत्य;
        }
        ध्रुव size_t शिखरम्_k = ids_pshape[ids_pshape.rank().get_दीर्घता() - 1].get_दीर्घता();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        ध्रुव स्वचालित routing_अन्तर् = pm.at(routing_m);
        ध्रुव स्वचालित routing_pshape = routing_अन्तर्.get_partial_shape();
        ध्रुव स्वचालित routing_rank = routing_pshape.rank();
        यदि (routing_rank.is_dynamic() || (routing_rank.get_दीर्घता() != 4 && routing_rank.get_दीर्घता() != 3) ||
            routing_pshape[routing_rank.get_दीर्घता() - 1] != 1) {
            प्रत्यागम असत्य;
        }
        ध्रुव तार्किक batchless = routing_rank.get_दीर्घता() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        स्वचालित अधस्तात्_scale = pm.at(अधस्तात्_scale_m);
        यदि (अधस्तात्_scale.get_partial_shape() != routing_pshape) {
            प्रत्यागम असत्य;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_अन्तर्, अधस्तात्_scale);
        यदि (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        यदि (ids_pshape.rank().get_दीर्घता() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        यदि (routing.get_partial_shape() != ids.get_partial_shape()) {
            प्रत्यागम असत्य;
        }

        ध्रुव size_t अधस्तात्_k = logical_k(अधस्तात्_shape);
        ध्रुव स्वचालित अधस्तात्_scale_shape = अधस्तात्.scale.get_shape();
        ध्रुव size_t अधस्तात्_groups = अधस्तात्_scale_shape.size() >= 3 ? अधस्तात्_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        ध्रुव ov::Output<ov::Node> gate_weight = खण्डिका_experts_dim1(gate_ऊर्ध्वम्.weight, 0, mid);
        ध्रुव ov::Output<ov::Node> ऊर्ध्वम्_weight = खण्डिका_experts_dim1(gate_ऊर्ध्वम्.weight, mid, पूर्णम्);
        ध्रुव ov::Output<ov::Node> gate_scale = खण्डिका_experts_dim1(gate_ऊर्ध्वम्.scale, 0, mid);
        ध्रुव ov::Output<ov::Node> ऊर्ध्वम्_scale = खण्डिका_experts_dim1(gate_ऊर्ध्वम्.scale, mid, पूर्णम्);
        ध्रुव ov::Output<ov::Node> gate_zp = खण्डिका_experts_dim1(gate_ऊर्ध्वम्.zp, 0, mid);
        ध्रुव ov::Output<ov::Node> ऊर्ध्वम्_zp = खण्डिका_experts_dim1(gate_ऊर्ध्वम्.zp, mid, पूर्णम्);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_आकार = logical_k(gate_ऊर्ध्वम्_shape);
        config.inter_आकार = static_cast<size_t>(mid);
        config.अङ्कः_expert = gate_ऊर्ध्वम्_shape[0];
        config.अङ्कः_shared_expert = 0;
        config.शिखरम्_k = शिखरम्_k;
        config.गणः_आकार = अधस्तात्_groups <= 1 ? std::numeric_limits<size_t>::max() : अधस्तात्_k / अधस्तात्_groups;
        config.has_batch_dim = सत्य;
        config.has_zp = सत्य;
        config.बहिर्गमनम्_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        ध्रुव स्वचालित a_transpose = pm.at(a_m).get_ग्रन्थिः_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->निवेशः_मानम्(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        ध्रुव ov::Output_सदिशम् तर्काः_2 = {
            hidden,      routing,     ids,
            gate_weight, gate_scale,  gate_zp,
            ऊर्ध्वम्_weight,   ऊर्ध्वम्_scale,    ऊर्ध्वम्_zp,
            अधस्तात्.weight, अधस्तात्.scale,  अधस्तात्.zp,
        };

        स्वचालित moe = std::make_shared<ov::op::internal::MOECompressed>(तर्काः_2, config);

        ov::Output<ov::Node> परिणाम = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        यदि (batchless) {
            परिणाम = std::make_shared<ov::op::v0::Squeeze>(
                परिणाम, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ध्रुव स्वचालित मूलम्_type = m.get_मेलनम्_मूलम्()->get_output_element_type(0);
        यदि (परिणाम.get_element_type() != मूलम्_type) {
            परिणाम = std::make_shared<ov::op::v0::Convert>(परिणाम, मूलम्_type);
        }
        यदि (परिणाम.get_partial_shape() != m.get_मेलनम्_मूलम्()->get_output_partial_shape(0)) {
            प्रत्यागम असत्य;
        }

        परिणाम.get_ग्रन्थिः_shared_ptr()->समुच्चयः_friendly_name(m.get_मेलनम्_मूलम्()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), परिणाम.get_ग्रन्थिः_shared_ptr());
        ov::replace_ग्रन्थिः(m.get_मेलनम्_मूलम्(), परिणाम.get_ग्रन्थिः_shared_ptr());
        पञ्जिकाकोशः_new_ग्रन्थिः(moe);
        प्रत्यागम सत्य;
    };

    पञ्जिकाकोशः_matcher(
        std::make_shared<Matcher>(मूलम्_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
