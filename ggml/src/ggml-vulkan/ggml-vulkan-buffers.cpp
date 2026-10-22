#ضمن "ggml-vulkan-common.h"

#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <mutex>
#ضمن <shared_mutex>

ساكن فراغ * portable_aligned_malloc(size_t align, size_t size) {
#جيڪڏهن_وصف_ٿيل _WIN32
    أرجع _aligned_malloc(size, align);
#وإلا
    فراغ * ptr = خالي_اشارو;
    إذا (posix_memalign(&ptr, align, size) != 0) {
        أرجع خالي_اشارو;
    }
    أرجع ptr;
#جيڪڏهن_پورو
}

ساكن فراغ portable_aligned_free(فراغ * ptr) {
#جيڪڏهن_وصف_ٿيل _WIN32
    _aligned_free(ptr);
#وإلا
    std::حرر(ptr);
#جيڪڏهن_پورو
}

ggml_backend_بفر_type_i ggml_backend_vk_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_vk_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_vk_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_بفر_type_get_max_سائيز,
    /* .get_alloc_size      = */ ggml_backend_vk_بفر_type_get_alloc_سائيز,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ساكن std::vector<uint32_t> ggml_vk_ڳوليو_يادگيري_properties(ثابت vk::Physical_Device_يادگيري_Properties* mem_props, vk::يادگيري_Requirements* mem_req, vk::يادگيري_Property_Flags flags) {
    std::vector<uint32_t> indices;

    لكل (uint32_t i = 0; i < mem_props->يادگيري_Type_ڳڻپ; ++i) {
        vk::يادگيري_Type يادگيري_type = mem_props->يادگيري_Types[i];
        إذا ((mem_req->يادگيري_Type_بٽ & ((uint64_t)1 << i)) &&
            (flags & يادگيري_type.propertyFlags) == flags &&
            mem_props->يادگيري_Heaps[يادگيري_type.ڍير_اشاريو].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    أرجع indices;
}

ساكن vk_بفر ggml_vk_create_بفر(vk_device& device, size_t size, ثابت std::initializer_list<vk::يادگيري_Property_Flags> & req_flags_list,
                                       فراغ *import_ptr = خالي_اشارو) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    إذا (size > device->max_بفر_سائيز) {
        ارم vk::ٻاهريون_جو_Device_يادگيري_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_بفر buf = std::make_shared<vk_بفر_struct>();

    إذا (size == 0) {
        buf->size = 0;
        أرجع buf;
    }

    vk::بفر_Usage_Flags usage_flags = vk::بفر_Usage_Flag_بٽ::e_Storage_بفر | vk::بفر_Usage_Flag_بٽ::eTransferSrc | vk::بفر_Usage_Flag_بٽ::eTransferDst;
    vk::يادگيري_Allocate_Flags mem_flags {};
    إذا (device->بفر_device_پتو) {
        usage_flags |= vk::بفر_Usage_Flag_بٽ::e_Shader_Device_پتو;
        mem_flags |= vk::يادگيري_Allocate_Flag_بٽ::e_Device_پتو;
    }

    vk::بفر_Create_Info بفر_create_info{
        vk::بفر_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_گهڻائي::eExclusive,
        0,
        خالي_اشارو,
    };

    vk::External_يادگيري_بفر_Create_Info external_يادگيري_bci;
    إذا (import_ptr) {
        external_يادگيري_bci.handleTypes = vk::External_يادگيري_Handle_Type_Flag_بٽ::e_Host_ورهاست_EXT;
        بفر_create_info.سيٽ_PNext(&external_يادگيري_bci);
    }

    buf->بفر = device->device.create_بفر(بفر_create_info);

    vk::يادگيري_Requirements mem_req = device->device.get_بفر_يادگيري_Requirements(buf->بفر);

    vk::Physical_Device_يادگيري_Properties mem_props = device->physical_device.get_يادگيري_Properties();

    ثابت vk::يادگيري_ترجيح_Allocate_Info_EXT mem_ترجيح_info { 1.0f };

    vk::يادگيري_Allocate_Flags_Info mem_flags_info { mem_flags };

    إذا (device->يادگيري_ترجيح) {
        mem_flags_info.سيٽ_PNext(&mem_ترجيح_info);
    }

    إذا (import_ptr) {
        vk::يادگيري_Host_اشارو_Properties_EXT host_اشارو_props;
        حاول {
            host_اشارو_props = device->device.get_يادگيري_Host_اشارو_Properties_EXT(vk::External_يادگيري_Handle_Type_Flag_بٽ::e_Host_ورهاست_EXT, import_ptr);
        } امسك (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_بفر(buf->بفر);
            أرجع {};
        }
        vk::Physical_Device_يادگيري_Properties mem_props = device->physical_device.get_يادگيري_Properties();

        uint32_t يادگيري_type_idx;
        vk::يادگيري_Property_Flags property_flags = *req_flags_list.begin();
        لكل (يادگيري_type_idx = 0; يادگيري_type_idx < 32; ++يادگيري_type_idx) {
            إذا (!(host_اشارو_props.يادگيري_Type_بٽ & (1u << يادگيري_type_idx))) {
                استمر;
            }
            إذا (!(mem_req.يادگيري_Type_بٽ & (1u << يادگيري_type_idx))) {
                استمر;
            }

            vk::يادگيري_Type يادگيري_type = mem_props.يادگيري_Types[يادگيري_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            إذا ((يادگيري_type.propertyFlags & property_flags) == property_flags) {
                property_flags = يادگيري_type.propertyFlags;
                اكسر;
            }
        }
        إذا (يادگيري_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_بفر(buf->بفر);
            أرجع {};
        }

        buf->يادگيري_property_flags = mem_props.يادگيري_Types[يادگيري_type_idx].propertyFlags;
        حاول {
            vk::Import_يادگيري_Host_اشارو_Info_EXT import_info;
            import_info.handleType = vk::External_يادگيري_Handle_Type_Flag_بٽ::e_Host_ورهاست_EXT;
            import_info.p_Host_اشارو = import_ptr;
            import_info.سيٽ_PNext(&mem_flags_info);
            buf->device_يادگيري = device->device.allocate_يادگيري({ size, يادگيري_type_idx, &import_info });
        } امسك (ثابت vk::SystemError& e) {
        }
    } وإلا {
        لكل (خودڪار it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ثابت خودڪار & req_flags = *it;

            ثابت std::vector<uint32_t> يادگيري_type_indices = ggml_vk_ڳوليو_يادگيري_properties(&mem_props, &mem_req, req_flags);

            إذا (يادگيري_type_indices.خالي_جايون()) {
                استمر;
            }

            منطقي پورو = خطأ;

            لكل (خودڪار mtype_it = يادگيري_type_indices.begin(); mtype_it != يادگيري_type_indices.end(); mtype_it++) {
                حاول {
                    buf->device_يادگيري = device->device.allocate_يادگيري({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->يادگيري_property_flags = mem_props.يادگيري_Types[*mtype_it].propertyFlags;
                    پورو = سچ;
                    اكسر;
                } امسك (ثابت vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    إذا (it + 1 == req_flags_list.end() && mtype_it + 1 == يادگيري_type_indices.end()) {
                        device->device.destroy_بفر(buf->بفر);
                        ارم e;
                    }
                }
            }

            إذا (پورو) {
                اكسر;
            }
        }
    }

    إذا (!buf->device_يادگيري) {
        device->device.destroy_بفر(buf->بفر);
        ارم vk::ٻاهريون_جو_Device_يادگيري_Error("No suitable memory type found");
    }

    buf->ptr = خالي_اشارو;

    إذا (import_ptr) {
        buf->ptr = import_ptr;
    } وإلا {
        إذا (buf->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostVisible) {
            buf->ptr = device->device.map_يادگيري(buf->device_يادگيري, 0, VK_سڄو_سائيز);
        }
    }

    device->device.bind_بفر_يادگيري(buf->بفر, buf->device_يادگيري, 0);

    buf->device = device;
    buf->size = size;

    إذا (device->بفر_device_پتو) {
        ثابت vk::بفر_Device_پتو_Info پتو_Info(buf->بفر);
        buf->bda_addr = device->device.get_بفر_پتو(پتو_Info);
    }

    device->يادگيري_logger->log_ورهاست(buf, size);

    أرجع buf;
}

vk_بفر ggml_vk_create_بفر_check(vk_device& device, size_t size, vk::يادگيري_Property_Flags req_flags, vk::يادگيري_Property_Flags fallback_flags) {
    حاول {
        أرجع ggml_vk_create_بفر(device, size, {req_flags, fallback_flags});
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::سٽ_پڄاڻي;
        std::cerr << "ggml_vulkan: " << e.what() << std::سٽ_پڄاڻي;
        ارم e;
    }
}

vk_بفر ggml_vk_create_بفر_device(vk_device& device, size_t size) {
    vk_بفر buf;
    حاول {
        إذا (device->prefer_host_يادگيري) {
            buf = ggml_vk_create_بفر(device, size, {vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent,
                                                       vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
        } وإلا إذا (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_بفر(device, size, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal | vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent,
                                                       vk::يادگيري_Property_Flag_بٽ::eDeviceLocal,
                                                       vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent});
        } وإلا إذا (device->disable_host_visible_vidmem) {
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal,
                                                           vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_بفر(device, size, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
            }
        } وإلا {
            // use rebar if available, otherwise fallback to device only visible memory
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal | vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent,
                                                           vk::يادگيري_Property_Flag_بٽ::eDeviceLocal,
                                                           vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_بفر(device, size, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal | vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent,
                                                           vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
            }
        }
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::سٽ_پڄاڻي;
        std::cerr << "ggml_vulkan: " << e.what() << std::سٽ_پڄاڻي;
        ارم e;
    }

    أرجع buf;
}

فراغ ggml_vk_destroy_بفر(vk_بفر& buf) {
    إذا (buf == خالي_اشارو) {
        أرجع;
    }

    إذا (buf->device != خالي_اشارو) {
        buf->device->يادگيري_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ساكن size_t ggml_vk_host_ورهاست_ڳوليو(ثابت فراغ * ptr) {
    لكل (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ثابت uint8_t * بنياد = (ثابت uint8_t *) vk_instance.host_allocations[i].بنياد;
        إذا (ptr >= بنياد && ptr < بنياد + vk_instance.host_allocations[i].size) {
            أرجع i;
        }
    }
    أرجع SIZE_MAX;
}

فراغ * ggml_vk_host_malloc(size_t size) {
    VK_LOG_يادگيري("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    إذا (vk_instance.multi_device) {
        ثابت size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ثابت size_t alloc_سائيز = (size + align - 1) / align * align;

        فراغ * ptr = portable_aligned_malloc(align, alloc_سائيز);
        إذا (ptr == خالي_اشارو) {
            أرجع خالي_اشارو;
        }
        vk_host_ورهاست داخلا;
        داخلا.بنياد = ptr;
        داخلا.size = alloc_سائيز;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(داخلا));
        أرجع ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_بفر buf;
    حاول {
        buf = ggml_vk_create_بفر(dev0, size,
            {vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent | vk::يادگيري_Property_Flag_بٽ::eHostCached,
             vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent});
    } امسك (vk::SystemError&) {
        أرجع خالي_اشارو;
    }

    إذا (!(buf->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_بفر(buf);
        أرجع خالي_اشارو;
    }

    vk_host_ورهاست داخلا;
    داخلا.بنياد             = buf->ptr;
    داخلا.size             = size;
    داخلا.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(داخلا));
    أرجع buf->ptr;
}

فراغ ggml_vk_host_free(فراغ * ptr) {
    إذا (ptr == خالي_اشارو) {
        أرجع;
    }
    VK_LOG_يادگيري("ggml_vk_host_free(" << ptr << ")");

    vk_host_ورهاست داخلا;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ثابت size_t اشاريو = ggml_vk_host_ورهاست_ڳوليو(ptr);
        إذا (اشاريو == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            أرجع;
        }
        داخلا = std::move(vk_instance.host_allocations[اشاريو]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + اشاريو);
    }

    لكل (خودڪار & view : داخلا.views) {
        ggml_vk_destroy_بفر(view);
    }
    إذا (vk_instance.multi_device) {
        portable_aligned_free(داخلا.بنياد);
    }
}

فراغ ggml_vk_host_get(ثابت vk_device& device, ثابت فراغ * ptr, vk_بفر& buf, size_t& buf_وٿي) {
    buf = خالي_اشارو;
    buf_وٿي = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ثابت size_t اشاريو = ggml_vk_host_ورهاست_ڳوليو(ptr);
    إذا (اشاريو == SIZE_MAX) {
        أرجع;
    }
    vk_host_ورهاست & داخلا = vk_instance.host_allocations[اشاريو];

    إذا (داخلا.views[device->idx] == خالي_اشارو) {
        ثابت uint32_t ٻٽو_انگ = 1u << device->idx;
        إذا (!vk_instance.multi_device || !vk_instance.device_host_بفر_import[device->idx] || (vk_instance.host_import_failed & ٻٽو_انگ)) {
            أرجع;
        }
        GGML_ASSERT(device->external_يادگيري_host);
        // const only propagates from the const context; the pointee is mutable
        داخلا.views[device->idx] = ggml_vk_بفر_from_host_ptr(const_cast<vk_device&>(device), داخلا.بنياد, داخلا.size);
        إذا (داخلا.views[device->idx] == خالي_اشارو) {
            vk_instance.host_import_failed |= ٻٽو_انگ;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            أرجع;
        }
    }

    buf = داخلا.views[device->idx];
    buf_وٿي = (ثابت uint8_t *) ptr - (ثابت uint8_t *) داخلا.بنياد;
}

فراغ ggml_vk_ensure_sync_staging_بفر(vk_device& device, size_t size) {
    إذا (device->sync_staging == خالي_اشارو || device->sync_staging->size < size) {
        VK_LOG_يادگيري("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(device->sync_staging);
        device->sync_staging = ggml_vk_create_بفر_check(device, size,
            vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent | vk::يادگيري_Property_Flag_بٽ::eHostCached,
            vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent);
    }
}

فراغ ggml_vk_ensure_sync_staging_بفر(ggml_backend_vk_context * ctx, size_t size) {
    إذا (ctx->sync_staging == خالي_اشارو || ctx->sync_staging->size < size) {
        VK_LOG_يادگيري("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_بفر_check(ctx->device, size,
            vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent | vk::يادگيري_Property_Flag_بٽ::eHostCached,
            vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent);
    }
}

ساكن فراغ ggml_vk_بفر_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_بفر& dst, size_t وٿي, ثابت ggml_tensor * tensor, منطقي sync_staging = خطأ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    إذا(dst->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::سٽ_پڄاڻي;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_بفر buf = خالي_اشارو;
    size_t buf_وٿي = 0;
    ggml_vk_host_get(ctx->device, tensor->معلومات, buf, buf_وٿي);

    ثابت uint64_t ne0 = tensor->ne[0];
    ثابت uint64_t ne1 = tensor->ne[1];
    ثابت uint64_t ne2 = tensor->ne[2];
    ثابت uint64_t ne3 = tensor->ne[3];
    ثابت uint64_t nb0 = tensor->nb[0];
    ثابت uint64_t nb1 = tensor->nb[1];
    ثابت uint64_t nb2 = tensor->nb[2];
    ثابت uint64_t nb3 = tensor->nb[3];
    ثابت ggml_type type = tensor->type;
    ثابت uint64_t ts = ggml_type_سائيز(type);
    ثابت uint64_t bs = ggml_blck_سائيز(type);

    ثابت uint64_t dstnb0 = ts;
    ثابت uint64_t dstnb1 = dstnb0*(ne0/bs);
    ثابت uint64_t dstnb2 = dstnb1*ne1;
    ثابت uint64_t dstnb3 = dstnb2*ne2;

    ثابت uint64_t ne = ggml_nelements(tensor);

    إذا (buf != خالي_اشارو) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices;

        لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
            لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                إذا (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_وٿي + i3*nb3 + i2*nb2, وٿي + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } وإلا {
                    لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                        إذا (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_وٿي + i3*nb3 + i2*nb2 + i1*nb1, وٿي + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } وإلا {
                            ثابت uint64_t s_off = buf_وٿي + i3*nb3 + i2*nb2 + i1*nb1;
                            ثابت uint64_t d_off = وٿي + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        أرجع;
    }

    إذا (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_بفر& staging = ctx->device->sync_staging;
    ثابت uint64_t copy_سائيز = ts*ne/bs;
    ggml_vk_ensure_sync_staging_بفر(ctx->device, copy_سائيز);
    Vk_بفر_Copy buf_copy{ 0, وٿي, copy_سائيز };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_بفر(subctx->s->بفر->buf, (Vk_بفر)staging->بفر, (Vk_بفر)dst->بفر, 1, &buf_copy);

    لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
        لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            إذا (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ثابت uint8_t *) tensor->معلومات + buf_وٿي + i3*nb3 + i2*nb2, dstnb2, &subctx->اندريون_memcpys);
            } وإلا {
                لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                    إذا (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ثابت uint8_t *) tensor->معلومات + buf_وٿي + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->اندريون_memcpys);
                    } وإلا {
                        ثابت uint64_t s_off = buf_وٿي + i3*nb3 + i2*nb2 + i1*nb1;
                        ثابت uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ثابت uint8_t *) tensor->معلومات + s_off + i0*nb0, dstnb0, &subctx->اندريون_memcpys);
                        }
                    }
                }
            }
        }
    }
}

منطقي ggml_vk_بفر_write_2d_async(vk_context subctx, vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << ويڪر << ", " << اوچائي << ")");
    // Check if src is pinned memory
    vk_بفر buf = خالي_اشارو;
    size_t buf_وٿي = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_وٿي);

    إذا (buf != خالي_اشارو) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices(1);
        إذا (ويڪر == spitch && ويڪر == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_وٿي_2 = buf_وٿي;
            slices[0].dst_وٿي_2 = وٿي;
            slices[0].size = ويڪر * اوچائي;
        } وإلا {
            slices.resize(اوچائي);
            لكل (size_t i = 0; i < اوچائي; i++) {
                slices[i].src_وٿي_2 = buf_وٿي + i * spitch;
                slices[i].dst_وٿي_2 = وٿي + i * dpitch;
                slices[i].size = ويڪر;
            }
        }

        ggml_vk_sync_buffers(خالي_اشارو, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        أرجع سچ;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Staging buffer required
    ثابت size_t staging_سائيز = ويڪر * اوچائي;
    ggml_vk_ensure_sync_staging_بفر(dst->device, staging_سائيز);

    vk_بفر& staging_بفر = dst->device->sync_staging;

    std::vector<vk::بفر_Copy> slices(1);
    إذا (ويڪر == dpitch) {
        slices[0].src_وٿي_2 = 0;
        slices[0].dst_وٿي_2 = وٿي;
        slices[0].size = staging_سائيز;
    } وإلا {
        slices.resize(اوچائي);
        لكل (size_t i = 0; i < اوچائي; i++) {
            slices[i].src_وٿي_2 = i * ويڪر;
            slices[i].dst_وٿي_2 = وٿي + i * dpitch;
            slices[i].size = ويڪر;
        }
    }

    ggml_vk_sync_buffers(خالي_اشارو, subctx);
    subctx->s->بفر->buf.copy_بفر(staging_بفر->بفر, dst->بفر, slices);

    إذا (ويڪر == spitch) {
        deferred_memcpy((uint8_t *)staging_بفر->ptr, src, staging_سائيز, &subctx->اندريون_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < اوچائي; i++) {
            deferred_memcpy((uint8_t *)staging_بفر->ptr + i * ويڪر, (ثابت uint8_t *) src + i * spitch, ويڪر, &subctx->اندريون_memcpys);
        }
    }
    أرجع سچ;
}

منطقي ggml_vk_بفر_write_async(vk_context subctx, vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t size, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    أرجع ggml_vk_بفر_write_2d_async(subctx, dst, وٿي, src, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_بفر_write_2d(vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << ويڪر << ", " << اوچائي << ")");
    // Buffer is already mapped
    إذا(dst->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostVisible) {
        GGML_ASSERT(dst->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostCoherent);

        إذا (ويڪر == spitch && ويڪر == dpitch) {
            يادگيري_نقل((uint8_t *)dst->ptr + وٿي, src, ويڪر * اوچائي);
        } وإلا {
            لكل (size_t i = 0; i < اوچائي; i++) {
                يادگيري_نقل((uint8_t *)dst->ptr + وٿي + i * dpitch, (ثابت uint8_t *) src + i * spitch, ويڪر);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        منطقي ret = ggml_vk_بفر_write_2d_async(subctx, dst, وٿي, src, spitch, dpitch, ويڪر, اوچائي, سچ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_پڇاڙي(subctx);

        لكل (خودڪار& cpy : subctx->اندريون_memcpys) {
            يادگيري_نقل(cpy.dst, cpy.src, cpy.n);
        }

        لكل (خودڪار& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, سچ, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_قطار_command_pools_cleanup(dst->device);
    }
}

فراغ ggml_vk_بفر_write(vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_بفر_write_2d(dst, وٿي, src, size, size, size, 1);
}

منطقي ggml_vk_بفر_read_2d_async(vk_context subctx, vk_بفر& src, size_t وٿي, فراغ * dst, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << وٿي << ", width=" << ويڪر << ", height=" << اوچائي << ")");
    GGML_ASSERT(ويڪر > 0);
    GGML_ASSERT(اوچائي > 0);
    GGML_ASSERT(src != خالي_اشارو);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_بفر buf = خالي_اشارو;
    size_t buf_وٿي = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_وٿي);

    std::vector<vk::بفر_Copy> slices(1);
    إذا (ويڪر == spitch && ويڪر == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_وٿي_2 = وٿي;
        slices[0].dst_وٿي_2 = buf_وٿي;
        slices[0].size = ويڪر * اوچائي;
    } وإلا {
        slices.resize(اوچائي);
        لكل (size_t i = 0; i < اوچائي; i++) {
            slices[i].src_وٿي_2 = وٿي + i * spitch;
            slices[i].dst_وٿي_2 = buf_وٿي + i * dpitch;
            slices[i].size = ويڪر;
        }
    }

    إذا (buf != خالي_اشارو) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(خالي_اشارو, subctx);
        subctx->s->بفر->buf.copy_بفر(src->بفر, buf->بفر, slices);

        أرجع سچ;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Fall back to staging buffer
    ثابت size_t staging_سائيز = ويڪر * اوچائي;
    ggml_vk_ensure_sync_staging_بفر(src->device, staging_سائيز);

    vk_بفر& staging_بفر = src->device->sync_staging;

    std::vector<vk::بفر_Copy> staging_slices(1);
    إذا (ويڪر == spitch) {
        staging_slices[0].src_وٿي_2 = وٿي;
        staging_slices[0].dst_وٿي_2 = 0;
        staging_slices[0].size = staging_سائيز;
    } وإلا {
        staging_slices.resize(اوچائي);
        لكل (size_t i = 0; i < اوچائي; i++) {
            staging_slices[i].src_وٿي_2 = وٿي + i * spitch;
            staging_slices[i].dst_وٿي_2 = i * ويڪر;
            staging_slices[i].size = ويڪر;
        }
    }

    ggml_vk_sync_buffers(خالي_اشارو, subctx);
    subctx->s->بفر->buf.copy_بفر(src->بفر, staging_بفر->بفر, staging_slices);

    إذا (ويڪر == dpitch) {
        deferred_memcpy(dst, staging_بفر->ptr, staging_سائيز, &subctx->ٻاهريون_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < اوچائي; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) staging_بفر->ptr + i * ويڪر, ويڪر, &subctx->ٻاهريون_memcpys);
        }
    }
    أرجع سچ;
}

ساكن منطقي ggml_vk_بفر_read_async(vk_context subctx, vk_بفر& src, size_t وٿي, فراغ * dst, size_t size, منطقي sync_staging = خطأ) {
    أرجع ggml_vk_بفر_read_2d_async(subctx, src, وٿي, dst, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_بفر_read_2d(vk_بفر& src, size_t وٿي, فراغ * dst, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->بفر << ", " << وٿي << ", " << ويڪر << ", " << اوچائي << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ثابت منطقي slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostCached);
    إذا(src->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->بفر->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_بٽ::eComputeShader | vk::Pipeline_Stage_Flag_بٽ::eTransfer,
            vk::Pipeline_Stage_Flag_بٽ::eHost,
            {},
            { { vk::Access_Flag_بٽ::eShaderWrite | vk::Access_Flag_بٽ::eTransferWrite,
                vk::Access_Flag_بٽ::eHostRead } },
            {}, {});
        ggml_vk_ctx_پڇاڙي(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);

        إذا (ويڪر == spitch && ويڪر == dpitch) {
            يادگيري_نقل(dst, (ثابت uint8_t *) src->ptr + وٿي, ويڪر * اوچائي);
        } وإلا {
            لكل (size_t i = 0; i < اوچائي; i++) {
                يادگيري_نقل((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) src->ptr + وٿي + i * spitch, ويڪر);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        منطقي ret = ggml_vk_بفر_read_2d_async(subctx, src, وٿي, dst, spitch, dpitch, ويڪر, اوچائي, سچ);
        GGML_ASSERT(ret);
        ggml_vk_ctx_پڇاڙي(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);

        لكل (خودڪار& cpy : subctx->ٻاهريون_memcpys) {
            يادگيري_نقل(cpy.dst, cpy.src, cpy.n);
        }
    }
}

فراغ ggml_vk_بفر_read(vk_بفر& src, size_t وٿي, فراغ * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->بفر << ", " << وٿي << ", " << size << ")");
    ggml_vk_بفر_read_2d(src, وٿي, dst, size, size, size, 1);
}

فراغ ggml_vk_بفر_copy_async(vk_context& ctx, vk_بفر& dst, size_t dst_وٿي, vk_بفر& src, size_t src_وٿي, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_بفر_Copy bc{ src_وٿي, dst_وٿي, size };

    vk_Cmd_Copy_بفر(ctx->s->بفر->buf, (Vk_بفر)src->بفر, (Vk_بفر)dst->بفر, 1, &bc);
}

فراغ ggml_vk_بفر_copy(vk_بفر& dst, size_t dst_وٿي, vk_بفر& src, size_t src_وٿي, size_t size) {
    إذا (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_قطار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_بفر_copy_async(subctx, dst, dst_وٿي, src, src_وٿي, size);
        ggml_vk_ctx_پڇاڙي(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, سچ, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_قطار_command_pools_cleanup(src->device);
    } وإلا {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_بفر(src->device, size);

        // Copy to src staging buffer
        ggml_vk_بفر_copy(src->device->sync_staging, 0, src, src_وٿي, size);
        // Copy to dst buffer
        ggml_vk_بفر_write(dst, dst_وٿي, src->device->sync_staging->ptr, size);
    }
}

فراغ ggml_vk_بفر_memset_async(vk_context& ctx, vk_بفر& dst, size_t وٿي, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << وٿي << ", " << c << ", " << size << ")");

    إذا (dst->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + وٿي, c, size, &ctx->memsets);
        أرجع;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->بفر->buf.fill_بفر(dst->بفر, وٿي, size, c);
}

فراغ ggml_vk_بفر_memset(vk_بفر& dst, size_t وٿي, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << وٿي << ", " << c << ", " << size << ")");

    إذا (dst->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + وٿي, c, size);
        أرجع;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_قطار->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->بفر->buf.fill_بفر(dst->بفر, وٿي, size, c);
    ggml_vk_ctx_پڇاڙي(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, سچ, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_قطار_command_pools_cleanup(dst->device);
}

ggml_backend_بفر_i ggml_backend_vk_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_vk_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_vk_بفر_get_بنياد,
    /* .init_tensor     = */ ggml_backend_vk_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_بفر_سيٽ_tensor,
    /* .get_tensor      = */ ggml_backend_vk_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_بفر_سيٽ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_بفر_clear,
    /* .reset           = */ عدم,
};

vk_بفر ggml_vk_بفر_from_host_ptr(vk_device & device, فراغ * ptr, size_t size) {
    إذا (!device->external_يادگيري_host) {
        أرجع {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    إذا (uptr & (device->min_imported_host_اشارو_alignment - 1)) {
        أرجع {};
    }
    إذا (size & (device->min_imported_host_اشارو_alignment - 1)) {
        أرجع {};
    }

    ثابت vk::يادگيري_Property_Flags property_flags = vk::يادگيري_Property_Flag_بٽ::eHostVisible | vk::يادگيري_Property_Flag_بٽ::eHostCoherent | vk::يادگيري_Property_Flag_بٽ::eHostCached;

    vk_بفر buf {};
    حاول {
        buf = ggml_vk_create_بفر(device, size, { property_flags }, ptr);
    } امسك (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    أرجع buf;
}

