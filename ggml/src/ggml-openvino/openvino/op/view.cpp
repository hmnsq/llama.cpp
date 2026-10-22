#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_view(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    اگر (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        واپس {context.get_داخلہ(0)};
    }

    اگر (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ثابت std::string & vname = context.get_name();
        اگر (vname.find("ffn_moe_weighted") != std::string::npos) {
            خودکار src_ps = context.get_داخلہ_shape(0);
            خودکار dst_ps = context.get_output_shape();
            اگر (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                خودکار sst = context.get_داخلہ_stride(0);
                خودکار dst = context.get_output_stride();
                size_t voff = context.get_output_op_انحراف();
                خودکار ss = src_ps.مقصد_shape();
                خودکار dd = dst_ps.مقصد_shape();
                ثابت size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ثابت خودکار اندر_rank = context.get_داخلہ(0).get_partial_shape().rank();
                ثابت صحیح axis_سرکاؤ =
                    اندر_rank.is_static() ? (صحیح) nd - (صحیح) اندر_rank.get_لمبائی() : 0;
                منطقی سرکاؤ_ok = axis_سرکاؤ >= 0 && (size_t) axis_سرکاؤ < nd;
                برائے (صحیح a = 0; a < axis_سرکاؤ && سرکاؤ_ok; ++a) {
                    سرکاؤ_ok = (ss[a] == 1 && dd[a] == 1);
                }
                اگر (سرکاؤ_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<منطقی> مستعمل(nd, جھوٹ);
                    std::vector<صحیح> dst_مقصد_src(nd, -1);
                    منطقی ok = سچ;
                    برائے (size_t d = 0; d < nd; ++d) {
                        اگر (dd[d] == 1) {
                            جاری;
                        }
                        صحیح ملا = -1;
                        برائے (size_t s = 0; s < nd; ++s) {
                            اگر (!مستعمل[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                ملا = (صحیح) s;
                                توڑو;
                            }
                        }
                        اگر (ملا < 0) {
                            ok = جھوٹ;
                            توڑو;
                        }
                        مستعمل[ملا] = سچ;
                        dst_مقصد_src[d] = ملا;
                    }
                    صحیح dropped = -1;
                    اگر (ok) {
                        برائے (size_t s = 0; s < nd; ++s) {
                            اگر (!مستعمل[s] && ss[s] > 1) {
                                اگر (dropped >= 0) {
                                    ok = جھوٹ;
                                    توڑو;
                                }
                                dropped = (صحیح) s;
                            }
                        }
                    }
                    اگر (ok && dropped >= 0) {
                        ثابت size_t dstr = sst[dropped];
                        ثابت int64_t dsz = (int64_t) ss[dropped];
                        اگر (dstr > 0 && voff % dstr == 0) {
                            ثابت int64_t sel = (int64_t) (voff / dstr);
                            اگر (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::قاش>(
                                    context.get_داخلہ(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_سرکاؤ}));
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
                                صحیح dst_ov_axis = (dyn != -1) ? ((صحیح) nd - 1 - (صحیح) dyn) : -1;
                                صحیح src_ov_axis = (dst_ov_axis >= axis_سرکاؤ && dst_ov_axis < (صحیح) nd)
                                                      ? dst_مقصد_src[dst_ov_axis]
                                                      : -1;
                                اگر (dst_ov_axis >= 0 && src_ov_axis >= axis_سرکاؤ) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    خودکار sl_shape = std::make_shared<ov::op::v3::Shape_کا>(sl, ov::element::i64);
                                    خودکار tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_سرکاؤ}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::Output_سمتیہ parts;
                                    برائے (صحیح a = axis_سرکاؤ; a < (صحیح) nd; ++a) {
                                        اگر (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } ورنہ {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    خودکار dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    خودکار rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, جھوٹ);
                                    واپس rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_سرکاؤ, dd.end());
                                خودکار dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                خودکار rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, جھوٹ);
                                واپس rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        واپس {context.get_داخلہ(0)};
    }

    خودکار داخلہ = context.get_داخلہ(0);
    خودکار src_shape = context.get_داخلہ_shape(0);
    خودکار dst_shape = context.get_output_shape();

    اگر (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        واپس {داخلہ};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    برائے (int64_t i = 0; i < src_shape.rank().get_لمبائی(); ++i) {
        اگر (src_shape[i].is_dynamic()) {
            واپس {داخلہ};
        }
        src_elems *= src_shape[i].get_لمبائی();
    }
    برائے (int64_t i = 0; i < dst_shape.rank().get_لمبائی(); ++i) {
        اگر (dst_shape[i].is_dynamic()) {
            واپس {داخلہ};
        }
        dst_elems *= dst_shape[i].get_لمبائی();
    }

    اگر (dst_elems >= src_elems) {
        واپس {داخلہ};
    }

    خودکار src_stride = context.get_داخلہ_stride(0);
    خودکار dst_stride = context.get_output_stride();
    size_t view_انحراف = context.get_output_op_انحراف();

    منطقی same_stride = (src_stride.size() == dst_stride.size());
    اگر (same_stride) {
        برائے (size_t i = 0; i < src_stride.size(); ++i) {
            اگر (src_stride[i] != dst_stride[i]) {
                same_stride = جھوٹ;
                توڑو;
            }
        }
    }

    اگر (!same_stride) {
        واپس {داخلہ};
    }

    خودکار src_ov_shape = src_shape.مقصد_shape();
    خودکار dst_ov_shape = dst_shape.مقصد_shape();
    size_t ndims = src_ov_shape.size();
    اگر (dst_ov_shape.size() != ndims) {
        واپس {داخلہ};
    }

    std::vector<صحیح> diff_dims;
    برائے (size_t i = 0; i < ndims; ++i) {
        اگر (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<صحیح>(i));
        }
    }

    اگر (diff_dims.size() != 1) {
        واپس {داخلہ};
    }

    صحیح قاش_dim = diff_dims[0];
    int64_t dim_سائز = static_cast<int64_t>(src_ov_shape[قاش_dim]);

    size_t ov_stride_for_dim = 1;
    برائے (size_t i = قاش_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_سائز = src_stride.back();
    اگر (elem_سائز == 0) {
        elem_سائز = 1;
    }

    int64_t begin_val = 0;
    اگر (ov_stride_for_dim > 0 && elem_سائز > 0) {
        begin_val = static_cast<int64_t>((view_انحراف / elem_سائز) / ov_stride_for_dim);
    }
    int64_t انتہا_val = begin_val + static_cast<int64_t>(dst_ov_shape[قاش_dim]);

    اگر (begin_val < 0 || انتہا_val > dim_سائز) {
        واپس {داخلہ};
    }

    خودکار sliced =
        std::make_shared<ov::op::v8::قاش>(داخلہ, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {انتہا_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {قاش_dim}));

    sliced->سیٹ_friendly_name(context.get_output_name());
    واپس {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
