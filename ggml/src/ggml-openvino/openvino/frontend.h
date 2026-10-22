// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#هدايت once

#ضمن <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_پڇاڙي {
عام:
    Front_پڇاڙي();

    ساكن std::shared_ptr<Model> convert(ثابت داخلا_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
