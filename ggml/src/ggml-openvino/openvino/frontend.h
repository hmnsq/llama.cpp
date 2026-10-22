// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#لارښوونه once

#ضمن <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_پای_ټکی {
عام:
    Front_پای_ټکی();

    ساكن std::shared_ptr<Model> convert(ثابت ننوت_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
