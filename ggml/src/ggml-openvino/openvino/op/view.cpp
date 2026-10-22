#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_view(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    إذا (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        أرجع {context.get_داخلا(0)};
    }

    إذا (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ثابت std::string & vname = context.get_name();
        إذا (vname.find("ffn_moe_weighted") != std::string::npos) {
            خودڪار src_ps = context.get_داخلا_shape(0);
            خودڪار dst_ps = context.get_output_shape();
            إذا (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                خودڪار sst = context.get_داخلا_stride(0);
                خودڪار dst = context.get_output_stride();
                size_t voff = context.get_output_op_وٿي();
                خودڪار ss = src_ps.منزل_shape();
                خودڪار dd = dst_ps.منزل_shape();
                ثابت size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ثابت خودڪار اندريون_rank = context.get_داخلا(0).get_partial_shape().rank();
                ثابت صحيح axis_سرڪايو =
                    اندريون_rank.is_static() ? (صحيح) nd - (صحيح) اندريون_rank.get_ڊيگهائي() : 0;
                منطقي سرڪايو_ok = axis_سرڪايو >= 0 && (size_t) axis_سرڪايو < nd;
                لكل (صحيح a = 0; a < axis_سرڪايو && سرڪايو_ok; ++a) {
                    سرڪايو_ok = (ss[a] == 1 && dd[a] == 1);
                }
                إذا (سرڪايو_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<منطقي> استعمال_ٿيل(nd, خطأ);
                    std::vector<صحيح> dst_منزل_src(nd, -1);
                    منطقي ok = سچ;
                    لكل (size_t d = 0; d < nd; ++d) {
                        إذا (dd[d] == 1) {
                            استمر;
                        }
                        صحيح مليو = -1;
                        لكل (size_t s = 0; s < nd; ++s) {
                            إذا (!استعمال_ٿيل[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                مليو = (صحيح) s;
                                اكسر;
                            }
                        }
                        إذا (مليو < 0) {
                            ok = خطأ;
                            اكسر;
                        }
                        استعمال_ٿيل[مليو] = سچ;
                        dst_منزل_src[d] = مليو;
                    }
                    صحيح dropped = -1;
                    إذا (ok) {
                        لكل (size_t s = 0; s < nd; ++s) {
                            إذا (!استعمال_ٿيل[s] && ss[s] > 1) {
                                إذا (dropped >= 0) {
                                    ok = خطأ;
                                    اكسر;
                                }
                                dropped = (صحيح) s;
                            }
                        }
                    }
                    إذا (ok && dropped >= 0) {
                        ثابت size_t dstr = sst[dropped];
                        ثابت int64_t dsz = (int64_t) ss[dropped];
                        إذا (dstr > 0 && voff % dstr == 0) {
                            ثابت int64_t sel = (int64_t) (voff / dstr);
                            إذا (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::ٽڪرو>(
                                    context.get_داخلا(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_سرڪايو}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                ثابت int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                صحيح dst_ov_axis = (dyn != -1) ? ((صحيح) nd - 1 - (صحيح) dyn) : -1;
                                صحيح src_ov_axis = (dst_ov_axis >= axis_سرڪايو && dst_ov_axis < (صحيح) nd)
                                                      ? dst_منزل_src[dst_ov_axis]
                                                      : -1;
                                إذا (dst_ov_axis >= 0 && src_ov_axis >= axis_سرڪايو) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    خودڪار sl_shape = std::make_shared<ov::op::v3::Shape_جو>(sl, ov::element::i64);
                                    خودڪار tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_سرڪايو}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::Output_ويڪٽر parts;
                                    لكل (صحيح a = axis_سرڪايو; a < (صحيح) nd; ++a) {
                                        إذا (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } وإلا {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    خودڪار dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    خودڪار rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, خطأ);
                                    أرجع rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_سرڪايو, dd.end());
                                خودڪار dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                خودڪار rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, خطأ);
                                أرجع rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        أرجع {context.get_داخلا(0)};
    }

    خودڪار داخلا_2 = context.get_داخلا(0);
    خودڪار src_shape = context.get_داخلا_shape(0);
    خودڪار dst_shape = context.get_output_shape();

    إذا (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        أرجع {داخلا_2};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    لكل (int64_t i = 0; i < src_shape.rank().get_ڊيگهائي(); ++i) {
        إذا (src_shape[i].is_dynamic()) {
            أرجع {داخلا_2};
        }
        src_elems *= src_shape[i].get_ڊيگهائي();
    }
    لكل (int64_t i = 0; i < dst_shape.rank().get_ڊيگهائي(); ++i) {
        إذا (dst_shape[i].is_dynamic()) {
            أرجع {داخلا_2};
        }
        dst_elems *= dst_shape[i].get_ڊيگهائي();
    }

    إذا (dst_elems >= src_elems) {
        أرجع {داخلا_2};
    }

    خودڪار src_stride = context.get_داخلا_stride(0);
    خودڪار dst_stride = context.get_output_stride();
    size_t view_وٿي = context.get_output_op_وٿي();

    منطقي same_stride = (src_stride.size() == dst_stride.size());
    إذا (same_stride) {
        لكل (size_t i = 0; i < src_stride.size(); ++i) {
            إذا (src_stride[i] != dst_stride[i]) {
                same_stride = خطأ;
                اكسر;
            }
        }
    }

    إذا (!same_stride) {
        أرجع {داخلا_2};
    }

    خودڪار src_ov_shape = src_shape.منزل_shape();
    خودڪار dst_ov_shape = dst_shape.منزل_shape();
    size_t ndims = src_ov_shape.size();
    إذا (dst_ov_shape.size() != ndims) {
        أرجع {داخلا_2};
    }

    std::vector<صحيح> diff_dims;
    لكل (size_t i = 0; i < ndims; ++i) {
        إذا (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<صحيح>(i));
        }
    }

    إذا (diff_dims.size() != 1) {
        أرجع {داخلا_2};
    }

    صحيح ٽڪرو_dim = diff_dims[0];
    int64_t dim_سائيز = static_cast<int64_t>(src_ov_shape[ٽڪرو_dim]);

    size_t ov_stride_for_dim = 1;
    لكل (size_t i = ٽڪرو_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_سائيز = src_stride.back();
    إذا (elem_سائيز == 0) {
        elem_سائيز = 1;
    }

    int64_t begin_val = 0;
    إذا (ov_stride_for_dim > 0 && elem_سائيز > 0) {
        begin_val = static_cast<int64_t>((view_وٿي / elem_سائيز) / ov_stride_for_dim);
    }
    int64_t پڇاڙي_val = begin_val + static_cast<int64_t>(dst_ov_shape[ٽڪرو_dim]);

    إذا (begin_val < 0 || پڇاڙي_val > dim_سائيز) {
        أرجع {داخلا_2};
    }

    خودڪار sliced =
        std::make_shared<ov::op::v8::ٽڪرو>(داخلا_2, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇاڙي_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ٽڪرو_dim}));

    sliced->سيٽ_friendly_name(context.get_output_name());
    أرجع {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
