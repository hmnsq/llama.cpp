#అన్తర్భావ "fuse_moe_compressed.h"

#అన్తర్భావ <cstring>
#అన్తర్భావ <limits>
#అన్తర్భావ <set>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/graph_util.hpp>
#అన్తర్భావ <openvino/core/rt_info.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/gelu.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reduce_sum.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <openvino/op/swish.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <openvino/pass/constant_folding.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/optional.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/wrap_type.hpp>

#అన్తర్భావ "../op/gather_matmul.hpp"
#అన్తర్భావ "../op/moe_compressed.hpp"

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

నామాకాశ {

నిర్మాణం dequant_inputs {
    ov::Output<ov::Node> బరువు;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zp;
    తార్కిక has_zp = అబద్ధం;
    తార్కిక ok = అబద్ధం;
};

// Peel the chain built by make_int4_weights/make_int8_weights back to its Constant inputs.
// Grouped weights keep the pre-Reshape rank-4 form [n_expert, n, k/group, group] with scale
// and zp at [n_expert, n, k/group, 1], which is the layout MOECompressed expects. Channel-wise
// weights stay rank-3 with a rank-3 scale and carry no zp.
dequant_inputs unwrap_dequant(స్థిరరాశి ov::Output<ov::Node> & b) {
    dequant_inputs res;

    స్వయంచాలక కణుపు = b.get_కణుపు_shared_ptr();
    అయితే (ov::is_type<ov::op::v0::Convert>(కణుపు) || ov::is_type<ov::op::v1::Reshape>(కణుపు)) {
        కణుపు = కణుపు->get_ఆదానం_కణుపు_shared_ptr(0);
    }

    స్వయంచాలక mul = ov::as_type_ptr<ov::op::v1::Multiply>(కణుపు);
    ఒకవేళ (!mul) {
        తిరిగి res;
    }
    res.scale = mul->ఆదానం_విలువ(1);

    స్వయంచాలక lhs = mul->get_ఆదానం_కణుపు_shared_ptr(0);
    ఒకవేళ (స్వయంచాలక sub = ov::as_type_ptr<ov::op::v1::వ్యవకలనం>(lhs)) {
        // Take the zero point down to its Constant: an integer zp is wrapped in a Convert to f16,
        // and the op wants the integer form. A natively quantized expert instead carries an exact
        // f16 zp (-min/scale) with no integer behind it, which the MoE kernel does not accept.
        స్వయంచాలక zp_కణుపు = sub->get_ఆదానం_కణుపు_shared_ptr(1);
        అయితే (ov::is_type<ov::op::v0::Convert>(zp_కణుపు)) {
            zp_కణుపు = zp_కణుపు->get_ఆదానం_కణుపు_shared_ptr(0);
        }
        res.zp = zp_కణుపు->output(0);
        res.has_zp = నిజం;
        lhs = sub->get_ఆదానం_కణుపు_shared_ptr(0);
    }
    అయితే (ov::is_type<ov::op::v0::Convert>(lhs)) {
        lhs = lhs->get_ఆదానం_కణుపు_shared_ptr(0);
    }
    ఒకవేళ (!ov::is_type<ov::op::v0::Constant>(lhs)) {
        తిరిగి res;
    }

    res.బరువు = lhs->output(0);
    res.ok = res.scale.get_partial_shape().is_static() && res.బరువు.get_partial_shape().is_static();
    తిరిగి res;
}

size_t logical_k(స్థిరరాశి ov::Shape & shape) {
    తిరిగి shape.size() == 4 ? shape[2] * shape[3] : shape.back();
}

// Slice a [n_expert, m, ...] compressed weight/scale/zp Constant along axis 1 (m), [begin, end).
// Built by copying raw bytes directly instead of a graph Slice op: CommonOptimizations rewrites
// v8::Slice into v1::StridedSlice for constant folding regardless of disable_constant_folding
// (the rewrite doesn't carry the marking over), and that reference evaluator crashes on
// sub-byte (u4/i4) element types. Each axis-1 "row" (the trailing dims) is confirmed
// byte-aligned here -- weight/scale/zp trailing sizes are always whole groups -- so a
// byte-range memcpy per row is exact for both regular and sub-byte element types.
ov::Output<ov::Node> ముక్క_experts_dim1(స్థిరరాశి ov::Output<ov::Node> & tensor, int64_t begin, int64_t end) {
    స్వయంచాలక constant = ov::as_type_ptr<ov::op::v0::Constant>(tensor.get_కణుపు_shared_ptr());
    OPENVINO_ASSERT(constant, "slice_experts_dim1 expects a Constant input");

    స్థిరరాశి స్వయంచాలక & shape = constant->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "slice_experts_dim1 expects rank >= 2");
    స్థిరరాశి స్వయంచాలక & type = constant->get_element_type();

    size_t అడ్డువరుస_elems = 1;
    కొరకు (size_t i = 2; i < shape.size(); ++i) {
        అడ్డువరుస_elems *= shape[i];
    }
    స్థిరరాశి size_t అడ్డువరుస_బిట్లు = అడ్డువరుస_elems * type.bitwidth();
    OPENVINO_ASSERT(అడ్డువరుస_బిట్లు % 8 == 0, "slice_experts_dim1: row is not byte-aligned");
    స్థిరరాశి size_t అడ్డువరుస_బైట్లు = అడ్డువరుస_బిట్లు / 8;

    ov::Shape బహిర్గతం_shape = shape;
    బహిర్గతం_shape[1] = static_cast<size_t>(end - begin);

    std::vector<uint8_t> బహిర్గతం_దత్తాంశం(shape[0] * static_cast<size_t>(end - begin) * అడ్డువరుస_బైట్లు);
    స్థిరరాశి స్వయంచాలక * src = static_cast<స్థిరరాశి uint8_t *>(constant->get_దత్తాంశం_ptr());
    స్థిరరాశి size_t src_అడ్డువరుస_బైట్లు = shape[1] * అడ్డువరుస_బైట్లు;
    కొరకు (size_t e = 0; e < shape[0]; ++e) {
        std::జ్ఞాపకం_నకలు(బహిర్గతం_దత్తాంశం.దత్తాంశం() + e * static_cast<size_t>(end - begin) * అడ్డువరుస_బైట్లు,
                   src + e * src_అడ్డువరుస_బైట్లు + static_cast<size_t>(begin) * అడ్డువరుస_బైట్లు,
                   static_cast<size_t>(end - begin) * అడ్డువరుస_బైట్లు);
    }

    తిరిగి std::make_shared<ov::op::v0::Constant>(type, బహిర్గతం_shape, బహిర్గతం_దత్తాంశం.దత్తాంశం());
}

}  // namespace

FuseMoeCompressed::FuseMoeCompressed() {
    ఉపయోగించి నామాకాశ ov::pass::pattern;

    // The gate and up projections each get their own Reshape/Transpose of the hidden state and
    // their own Reshape of the routing ids, so every branch needs its own sub-pattern. On GPU
    // mul_mat_id also converts the activations to f16 before the op and back to f32 after it,
    // so those Converts are matched as optional.
    స్వయంచాలక hidden_gate_m = any_ఆదానం();
    స్వయంచాలక a_gate_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_gate_m, any_ఆదానం() });
    స్వయంచాలక a_gate_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_gate_reshape_m }), any_ఆదానం() });
    స్వయంచాలక hidden_పైకి_m = any_ఆదానం();
    స్వయంచాలక a_పైకి_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ hidden_పైకి_m, any_ఆదానం() }) }),
          any_ఆదానం() });

    స్వయంచాలక gate_w_m = any_ఆదానం();
    స్వయంచాలక పైకి_w_m = any_ఆదానం();
    స్వయంచాలక కిందికి_w_m = any_ఆదానం();
    స్వయంచాలక ids_gate_m = any_ఆదానం();
    స్వయంచాలక ids_పైకి_m = any_ఆదానం();
    స్వయంచాలక ids_కిందికి_m = any_ఆదానం();

    స్వయంచాలక bgm_gate_m = wrap_type<ov::op::internal::GatherMatmul>({ a_gate_m, gate_w_m, ids_gate_m, any_ఆదానం() });
    స్వయంచాలక gate_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_gate_m, any_ఆదానం() }), any_ఆదానం() }) });

    స్వయంచాలక silu_m = wrap_type<ov::op::v4::Swish>({ gate_u_m });

    స్వయంచాలక bgm_పైకి_m = wrap_type<ov::op::internal::GatherMatmul>({ a_పైకి_m, పైకి_w_m, ids_పైకి_m, any_ఆదానం() });
    స్వయంచాలక పైకి_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_పైకి_m, any_ఆదానం() }), any_ఆదానం() }) });
    స్వయంచాలక swiglu_m = wrap_type<ov::op::v1::Multiply>({ silu_m, పైకి_u_m });

    స్వయంచాలక d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ swiglu_m, any_ఆదానం() }) }),
          any_ఆదానం() });
    స్వయంచాలక bgm_కిందికి_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, కిందికి_w_m, ids_కిందికి_m, any_ఆదానం() });
    స్వయంచాలక కిందికి_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
        { wrap_type<ov::op::v1::Transpose>({ bgm_కిందికి_m, any_ఆదానం() }), any_ఆదానం() }) });

    స్వయంచాలక routing_m = any_ఆదానం();
    స్వయంచాలక weighted_m = wrap_type<ov::op::v1::Multiply>({ కిందికి_u_m, routing_m });
    స్వయంచాలక మూలం_m = wrap_type<ov::op::v1::Reduce_కూడిక>({ weighted_m, any_ఆదానం() });

    స్థిరరాశి స్వయంచాలక callback = [=](Matcher & m) {
        స్వయంచాలక & pm = m.get_నమూనా_విలువ_map();

        స్థిరరాశి స్వయంచాలక gate = unwrap_dequant(pm.at(gate_w_m));
        స్థిరరాశి స్వయంచాలక పైకి = unwrap_dequant(pm.at(పైకి_w_m));
        స్థిరరాశి స్వయంచాలక కిందికి = unwrap_dequant(pm.at(కిందికి_w_m));
        ఒకవేళ (!gate.ok || !పైకి.ok || !కిందికి.ok) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి స్వయంచాలక gate_shape = gate.బరువు.get_shape();
        స్థిరరాశి స్వయంచాలక పైకి_shape = పైకి.బరువు.get_shape();
        స్థిరరాశి స్వయంచాలక కిందికి_shape = కిందికి.బరువు.get_shape();
        ఒకవేళ (gate_shape != పైకి_shape || gate_shape.size() < 3 || కిందికి_shape.size() < 3) {
            తిరిగి అబద్ధం;
        }

        // MOECompressed carries one group_size and one has_zp for all three projections, so a
        // model whose down-proj is quantized differently from gate/up cannot be described. This
        // happens when ggml requantizes Q5_K/Q6_K experts to channel-wise int8.
        ఒకవేళ (gate.has_zp != కిందికి.has_zp || gate_shape.size() != కిందికి_shape.size()) {
            తిరిగి అబద్ధం;
        }

        // The kernel only takes an integer zero point (moe_3gemm_swiglu_opt validate_impl).
        ఒకవేళ (gate.has_zp) {
            స్థిర స్థిరరాశి std::set<ov::element::Type> int_zp_types = { ov::element::u4, ov::element::i4,
                                                                     ov::element::u8, ov::element::i8 };
            ఒకవేళ (int_zp_types.count(gate.zp.get_element_type()) == 0 ||
                int_zp_types.count(కిందికి.zp.get_element_type()) == 0) {
                తిరిగి అబద్ధం;
            }
        }

        // Config holds a single group_size for all three projections.
        స్థిరరాశి స్వయంచాలక సమూహం_యొక్క = [](స్థిరరాశి dequant_inputs & w) {
            స్థిరరాశి స్వయంచాలక s = w.బరువు.get_shape();
            తిరిగి s.size() == 4 ? s[3] : logical_k(s);
        };
        ఒకవేళ (సమూహం_యొక్క(gate) != సమూహం_యొక్క(పైకి) || సమూహం_యొక్క(gate) != సమూహం_యొక్క(కిందికి)) {
            తిరిగి అబద్ధం;
        }

        // all three branches must route the same hidden state through the same experts
        ఒకవేళ (pm.at(hidden_gate_m) != pm.at(hidden_పైకి_m)) {
            తిరిగి అబద్ధం;
        }

        స్వయంచాలక ids = pm.at(ids_కిందికి_m);
        స్థిరరాశి స్వయంచాలక ids_pshape = ids.get_partial_shape();
        ఒకవేళ (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_నిడివి() - 1].is_dynamic()) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి size_t శిఖరం_k = ids_pshape[ids_pshape.rank().get_నిడివి() - 1].get_నిడివి();

        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
        స్వయంచాలక routing = pm.at(routing_m);
        స్థిరరాశి స్వయంచాలక routing_pshape = routing.get_partial_shape();
        ఒకవేళ (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_నిడివి() != 4 ||
            routing_pshape[3] != 1) {
            తిరిగి అబద్ధం;
        }
        // MOE requires routing weights and ids to have the same shape. Drop the trailing 1 of the
        // routing weights and give the ids the leading batch dim, so both become [1, n_tokens, top_k].
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        ఒకవేళ (ids_pshape.rank().get_నిడివి() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ఒకవేళ (routing.get_partial_shape() != ids.get_partial_shape()) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి size_t కిందికి_k = logical_k(కిందికి_shape);
        స్థిరరాశి స్వయంచాలక కిందికి_scale_shape = కిందికి.scale.get_shape();
        స్థిరరాశి size_t కిందికి_groups = కిందికి_scale_shape.size() >= 3 ? కిందికి_scale_shape[2] : 1;

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_కొలత = logical_k(gate_shape);
        config.inter_కొలత = gate_shape[1];
        config.అంకె_expert = gate_shape[0];
        config.అంకె_shared_expert = 0;
        config.శిఖరం_k = శిఖరం_k;
        config.సమూహం_కొలత = కిందికి_groups <= 1 ? std::numeric_limits<size_t>::max() : కిందికి_k / కిందికి_groups;
        config.has_batch_dim = నిజం;
        config.has_zp = gate.has_zp;
        // dynamic makes the output follow the hidden state, so the plugin can lower this region
        // to f16 together with the rest of the graph
        config.బహిర్గతం_type = ov::element::dynamic;

        స్వయంచాలక absent_zp = [] {
            స్వయంచాలక zp = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{ 0 });
            ov::pass::disable_constant_folding(zp);
            తిరిగి zp->output(0);
        };

        // MOE takes its output type from the hidden state. Transpose the activations before the
        // f16 Convert that mul_mat_id adds on GPU, so the op stays f32 like the block it replaces
        // and the plugin can lower the whole region uniformly.
        స్థిరరాశి స్వయంచాలక a_transpose = pm.at(a_gate_m).get_కణుపు_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_gate_reshape_m), a_transpose->ఆదానం_విలువ(1));

        స్థిరరాశి ov::Output_సదిశ వాదనలు_2 = {
            hidden, routing, ids,
            gate.బరువు,  gate.scale,   gate.has_zp ? gate.zp : absent_zp(),
            పైకి.బరువు,    పైకి.scale,     పైకి.has_zp ? పైకి.zp : absent_zp(),
            కిందికి.బరువు,  కిందికి.scale,   కిందికి.has_zp ? కిందికి.zp : absent_zp(),
        };

        స్వయంచాలక moe = std::make_shared<ov::op::internal::MOECompressed>(వాదనలు_2, config);

        // MOE takes its output type from the hidden state, which is f16 on GPU, while the rest of
        // the ggml graph works in f32.
        ov::Output<ov::Node> ఫలితం = moe->output(0);
        స్థిరరాశి స్వయంచాలక మూలం_type = m.get_సరిపోలిక_మూలం()->get_output_element_type(0);
        ఒకవేళ (ఫలితం.get_element_type() != మూలం_type) {
            ఫలితం = std::make_shared<ov::op::v0::Convert>(ఫలితం, మూలం_type);
        }

        ఫలితం.get_కణుపు_shared_ptr()->సమితి_friendly_name(m.get_సరిపోలిక_మూలం()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), ఫలితం.get_కణుపు_shared_ptr());
        ov::replace_కణుపు(m.get_సరిపోలిక_మూలం(), ఫలితం.get_కణుపు_shared_ptr());
        రిజిస్టరు_new_కణుపు(moe);
        తిరిగి నిజం;
    };

    రిజిస్టరు_matcher(std::make_shared<Matcher>(మూలం_m, "ov::frontend::ggml::pass::FuseMoeCompressed"), callback);
}

Fuse_Moe_Compressed_Fused_Gate_పైకి::Fuse_Moe_Compressed_Fused_Gate_పైకి() {
    ఉపయోగించి నామాకాశ ov::pass::pattern;

    // A single GatherMatmul computes the fused gate+up projection; the gate/up split happens
    // AFTER the GEMM via two Slice ops on the last (feature) axis (see get_glu_inputs in
    // op/glu_geglu.cpp), unlike FuseMoeCompressed's two-separate-GatherMatmul models.
    స్వయంచాలక hidden_m = any_ఆదానం();
    స్వయంచాలక a_reshape_m = wrap_type<ov::op::v1::Reshape>({ hidden_m, any_ఆదానం() });
    స్వయంచాలక a_m = wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ a_reshape_m }), any_ఆదానం() });

    స్వయంచాలక gate_పైకి_w_m = any_ఆదానం();
    స్వయంచాలక ids_gate_పైకి_m = any_ఆదానం();
    స్వయంచాలక bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_పైకి_w_m, ids_gate_పైకి_m, any_ఆదానం() });
    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
    // Unsqueeze after the Transpose and may Convert afterwards:
    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
    // the other side of the Transpose:
    //     GatherMatmul -> Convert -> Transpose -> Slice
    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
    స్వయంచాలక gu_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_ఆదానం() });
    స్వయంచాలక gu_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_ఆదానం() }) });

    స్వయంచాలక gate_ముక్క_m = wrap_type<ov::op::v8::ముక్క>({ gu_u_m, any_ఆదానం(), any_ఆదానం(), any_ఆదానం(), any_ఆదానం() });
    స్వయంచాలక పైకి_ముక్క_m = wrap_type<ov::op::v8::ముక్క>({ gu_u_m, any_ఆదానం(), any_ఆదానం(), any_ఆదానం(), any_ఆదానం() });
    స్వయంచాలక gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_ముక్క_m });
    స్వయంచాలక geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, పైకి_ముక్క_m });

    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
    // into the Reshape.
    స్వయంచాలక d_t_m = wrap_type<ov::op::v1::Transpose>(
        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_ఆదానం() }), any_ఆదానం() }) }),
          any_ఆదానం() });
    స్వయంచాలక కిందికి_w_m = any_ఆదానం();
    స్వయంచాలక ids_కిందికి_m = any_ఆదానం();
    స్వయంచాలక bgm_కిందికి_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, కిందికి_w_m, ids_కిందికి_m, any_ఆదానం() });
    // Same two shapes as the gate/up branch above.
    స్వయంచాలక కిందికి_t_m =
        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_కిందికి_m }), any_ఆదానం() });
    స్వయంచాలక కిందికి_u_m =
        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ కిందికి_t_m, any_ఆదానం() }) });

    // gemma-4 applies an extra per-expert output scale to the down projection before the
    // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
    // Qwen-shaped models have no such scale, so this node is specific to this pattern.
    స్వయంచాలక కిందికి_scale_m = any_ఆదానం();
    స్వయంచాలక కిందికి_scaled_m = wrap_type<ov::op::v1::Multiply>({ కిందికి_u_m, కిందికి_scale_m });

    స్వయంచాలక routing_m = any_ఆదానం();
    స్వయంచాలక weighted_m = wrap_type<ov::op::v1::Multiply>({ కిందికి_scaled_m, routing_m });
    స్వయంచాలక మూలం_m = wrap_type<ov::op::v1::Reduce_కూడిక>({ weighted_m, any_ఆదానం() });

    స్థిరరాశి స్వయంచాలక callback = [=](Matcher & m) {
        స్వయంచాలక & pm = m.get_నమూనా_విలువ_map();

        స్థిరరాశి స్వయంచాలక gate_పైకి = unwrap_dequant(pm.at(gate_పైకి_w_m));
        స్థిరరాశి స్వయంచాలక కిందికి = unwrap_dequant(pm.at(కిందికి_w_m));
        ఒకవేళ (!gate_పైకి.ok || !కిందికి.ok) {
            తిరిగి అబద్ధం;
        }

        // oneDNN's weight-decompression GEMM needs a real zero point, so a symmetric (no zp)
        // projection cannot be fused here -- e.g. the GPU-only Q4_0 fallback in
        // ggml_openvino_get_extracted_layout for grouped 8-bit expert weights.
        ఒకవేళ (!gate_పైకి.has_zp || !కిందికి.has_zp) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి స్వయంచాలక gate_పైకి_shape = gate_పైకి.బరువు.get_shape();
        స్థిరరాశి స్వయంచాలక కిందికి_shape = కిందికి.బరువు.get_shape();
        ఒకవేళ (gate_పైకి_shape.size() < 3 || కిందికి_shape.size() < 3 || gate_పైకి_shape.size() != కిందికి_shape.size()) {
            తిరిగి అబద్ధం;
        }

        // The op reads the zero point straight off a weight port, so it must already be an
        // integer Constant. Requantized experts (GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all)
        // are; a native Q4_K expert keeps an exact f16 zp and is left to the unfused path
        // rather than rounded here -- rounding would put a live node on that port and the
        // expert GEMM would read the wrong zero point.
        స్థిర స్థిరరాశి std::set<ov::element::Type> int_types = { ov::element::u4, ov::element::i4,
                                                               ov::element::u8, ov::element::i8 };
        ఒకవేళ (int_types.count(gate_పైకి.బరువు.get_element_type()) == 0 ||
            int_types.count(కిందికి.బరువు.get_element_type()) == 0 ||
            int_types.count(gate_పైకి.zp.get_element_type()) == 0 ||
            int_types.count(కిందికి.zp.get_element_type()) == 0) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి స్వయంచాలక సమూహం_యొక్క = [](స్థిరరాశి dequant_inputs & w) {
            స్థిరరాశి స్వయంచాలక s = w.బరువు.get_shape();
            తిరిగి s.size() == 4 ? s[3] : logical_k(s);
        };
        ఒకవేళ (సమూహం_యొక్క(gate_పైకి) != సమూహం_యొక్క(కిందికి)) {
            తిరిగి అబద్ధం;
        }

        స్వయంచాలక gelu_కణుపు = ov::as_type_ptr<ov::op::v7::Gelu>(pm.at(gelu_m).get_కణుపు_shared_ptr());
        ఒకవేళ (!gelu_కణుపు || gelu_కణుపు->get_approximation_బహుళకం() != ov::op::Gelu_Approximation_బహుళకం::ERF) {
            తిరిగి అబద్ధం;
        }

        // Read the actual gate/up split point from the matched Slice nodes instead of assuming
        // inter_size/2, so this stays correct if the model's ffn_dim convention ever changes.
        స్వయంచాలక gate_ముక్క = ov::as_type_ptr<ov::op::v8::ముక్క>(pm.at(gate_ముక్క_m).get_కణుపు_shared_ptr());
        స్వయంచాలక పైకి_ముక్క = ov::as_type_ptr<ov::op::v8::ముక్క>(pm.at(పైకి_ముక్క_m).get_కణుపు_shared_ptr());
        స్వయంచాలక gate_చివర_c = ov::as_type_ptr<ov::op::v0::Constant>(gate_ముక్క->ఆదానం_విలువ(2).get_కణుపు_shared_ptr());
        స్వయంచాలక పైకి_begin_c = ov::as_type_ptr<ov::op::v0::Constant>(పైకి_ముక్క->ఆదానం_విలువ(1).get_కణుపు_shared_ptr());
        ఒకవేళ (!gate_చివర_c || !పైకి_begin_c) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి స్వయంచాలక gate_చివర = gate_చివర_c->cast_సదిశ<int64_t>().at(0);
        స్థిరరాశి స్వయంచాలక పైకి_begin = పైకి_begin_c->cast_సదిశ<int64_t>().at(0);
        ఒకవేళ (gate_చివర != పైకి_begin || gate_చివర <= 0 || gate_చివర >= static_cast<int64_t>(gate_పైకి_shape[1])) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి int64_t mid = gate_చివర;
        స్థిరరాశి int64_t నిండిన = static_cast<int64_t>(gate_పైకి_shape[1]);

        స్వయంచాలక ids = pm.at(ids_కిందికి_m);
        స్థిరరాశి స్వయంచాలక ids_pshape = ids.get_partial_shape();
        ఒకవేళ (ids_pshape.rank().is_dynamic() || ids_pshape[ids_pshape.rank().get_నిడివి() - 1].is_dynamic()) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి size_t శిఖరం_k = ids_pshape[ids_pshape.rank().get_నిడివి() - 1].get_నిడివి();

        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
        // below - and the op's own config - stays in the shape that is already validated on the
        // stateless path; the batch dim is taken back off the result at the end.
        స్థిరరాశి స్వయంచాలక routing_లోపల = pm.at(routing_m);
        స్థిరరాశి స్వయంచాలక routing_pshape = routing_లోపల.get_partial_shape();
        స్థిరరాశి స్వయంచాలక routing_rank = routing_pshape.rank();
        ఒకవేళ (routing_rank.is_dynamic() || (routing_rank.get_నిడివి() != 4 && routing_rank.get_నిడివి() != 3) ||
            routing_pshape[routing_rank.get_నిడివి() - 1] != 1) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి తార్కిక batchless = routing_rank.get_నిడివి() == 3;

        // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
        // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
        // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
        స్వయంచాలక కిందికి_scale = pm.at(కిందికి_scale_m);
        ఒకవేళ (కిందికి_scale.get_partial_shape() != routing_pshape) {
            తిరిగి అబద్ధం;
        }
        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_లోపల, కిందికి_scale);
        ఒకవేళ (batchless) {
            routing = std::make_shared<ov::op::v0::Unsqueeze>(
                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        routing = std::make_shared<ov::op::v0::Squeeze>(
            routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
        ఒకవేళ (ids_pshape.rank().get_నిడివి() == 2) {
            ids = std::make_shared<ov::op::v0::Unsqueeze>(
                ids, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        ఒకవేళ (routing.get_partial_shape() != ids.get_partial_shape()) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి size_t కిందికి_k = logical_k(కిందికి_shape);
        స్థిరరాశి స్వయంచాలక కిందికి_scale_shape = కిందికి.scale.get_shape();
        స్థిరరాశి size_t కిందికి_groups = కిందికి_scale_shape.size() >= 3 ? కిందికి_scale_shape[2] : 1;

        // Split the fused gate_up weight/scale/zp on the output axis. slice_experts_dim1 copies
        // raw bytes out of the Constant, so each half stays a Constant -- which is what the op
        // needs on its weight ports.
        స్థిరరాశి ov::Output<ov::Node> gate_బరువు = ముక్క_experts_dim1(gate_పైకి.బరువు, 0, mid);
        స్థిరరాశి ov::Output<ov::Node> పైకి_బరువు = ముక్క_experts_dim1(gate_పైకి.బరువు, mid, నిండిన);
        స్థిరరాశి ov::Output<ov::Node> gate_scale = ముక్క_experts_dim1(gate_పైకి.scale, 0, mid);
        స్థిరరాశి ov::Output<ov::Node> పైకి_scale = ముక్క_experts_dim1(gate_పైకి.scale, mid, నిండిన);
        స్థిరరాశి ov::Output<ov::Node> gate_zp = ముక్క_experts_dim1(gate_పైకి.zp, 0, mid);
        స్థిరరాశి ov::Output<ov::Node> పైకి_zp = ముక్క_experts_dim1(gate_పైకి.zp, mid, నిండిన);

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
        config.expert_alpha = 0.0f;
        config.expert_beta = 1.0f;
        config.gate_idx = 0;
        config.hidden_కొలత = logical_k(gate_పైకి_shape);
        config.inter_కొలత = static_cast<size_t>(mid);
        config.అంకె_expert = gate_పైకి_shape[0];
        config.అంకె_shared_expert = 0;
        config.శిఖరం_k = శిఖరం_k;
        config.సమూహం_కొలత = కిందికి_groups <= 1 ? std::numeric_limits<size_t>::max() : కిందికి_k / కిందికి_groups;
        config.has_batch_dim = నిజం;
        config.has_zp = నిజం;
        config.బహిర్గతం_type = ov::element::dynamic;

        // Rebuild the activations Transpose before mul_mat_id's f16 GPU Convert, so the op
        // stays f32 like the block it replaces (mirrors FuseMoeCompressed).
        స్థిరరాశి స్వయంచాలక a_transpose = pm.at(a_m).get_కణుపు_shared_ptr();
        ov::Output<ov::Node> hidden =
            std::make_shared<ov::op::v1::Transpose>(pm.at(a_reshape_m), a_transpose->ఆదానం_విలువ(1));

        // w0 is the activated (gate) lane and w1 the multiplied (up) lane -- see
        // moe_3gemm_swiglu_opt.cpp: "scratch.up = up(x) * silu(gate(x))".
        స్థిరరాశి ov::Output_సదిశ వాదనలు_2 = {
            hidden,      routing,     ids,
            gate_బరువు, gate_scale,  gate_zp,
            పైకి_బరువు,   పైకి_scale,    పైకి_zp,
            కిందికి.బరువు, కిందికి.scale,  కిందికి.zp,
        };

        స్వయంచాలక moe = std::make_shared<ov::op::internal::MOECompressed>(వాదనలు_2, config);

        ov::Output<ov::Node> ఫలితం = moe->output(0);
        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
        ఒకవేళ (batchless) {
            ఫలితం = std::make_shared<ov::op::v0::Squeeze>(
                ఫలితం, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
        }
        స్థిరరాశి స్వయంచాలక మూలం_type = m.get_సరిపోలిక_మూలం()->get_output_element_type(0);
        ఒకవేళ (ఫలితం.get_element_type() != మూలం_type) {
            ఫలితం = std::make_shared<ov::op::v0::Convert>(ఫలితం, మూలం_type);
        }
        ఒకవేళ (ఫలితం.get_partial_shape() != m.get_సరిపోలిక_మూలం()->get_output_partial_shape(0)) {
            తిరిగి అబద్ధం;
        }

        ఫలితం.get_కణుపు_shared_ptr()->సమితి_friendly_name(m.get_సరిపోలిక_మూలం()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), ఫలితం.get_కణుపు_shared_ptr());
        ov::replace_కణుపు(m.get_సరిపోలిక_మూలం(), ఫలితం.get_కణుపు_shared_ptr());
        రిజిస్టరు_new_కణుపు(moe);
        తిరిగి నిజం;
    };

    రిజిస్టరు_matcher(
        std::make_shared<Matcher>(మూలం_m, "ov::frontend::ggml::pass::FuseMoeCompressedFusedGateUp"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
