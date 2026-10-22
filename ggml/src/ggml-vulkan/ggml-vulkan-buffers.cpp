#অন্তর্ভাব "ggml-vulkan-common.h"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <shared_mutex>

স্থির শূন্য * portable_aligned_malloc(size_t align, size_t size) {
#ifdef _WIN32
    ফেরত _aligned_malloc(size, align);
#অন্যথা
    শূন্য * ptr = nullptr;
    যদি (posix_memalign(&ptr, align, size) != 0) {
        ফেরত nullptr;
    }
    ফেরত ptr;
#endif
}

স্থির শূন্য portable_aligned_free(শূন্য * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#অন্যথা
    std::মুক্ত(ptr);
#endif
}

ggml_backend_বাফার_type_i ggml_backend_vk_বাফার_type_interface = {
    /* .get_name            = */ ggml_backend_vk_বাফার_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_বাফার_type_alloc_বাফার,
    /* .alloc_buffer_n      = */ ফাঁকা,
    /* .get_alignment       = */ ggml_backend_vk_বাফার_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_বাফার_type_get_max_আকার,
    /* .get_alloc_size      = */ ggml_backend_vk_বাফার_type_get_alloc_আকার,
    /* .get_alloc_size_n    = */ ফাঁকা,
    /* .is_host             = */ ফাঁকা,
};

স্থির std::vector<uint32_t> ggml_vk_find_memory_properties(ধ্রুবক vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    জন্য (uint32_t i = 0; i < mem_props->memory_Type_গুনতি; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        যদি ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heap_সূচক].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    ফেরত indices;
}

স্থির vk_বাফার ggml_vk_create_বাফার(vk_device& device, size_t size, ধ্রুবক std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       শূন্য *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    যদি (size > device->max_বাফার_আকার) {
        ছোড়ো vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_বাফার buf = std::make_shared<vk_বাফার_struct>();

    যদি (size == 0) {
        buf->size = 0;
        ফেরত buf;
    }

    vk::বাফার_Usage_Flags usage_flags = vk::বাফার_Usage_Flag_Bits::e_Storage_বাফার | vk::বাফার_Usage_Flag_Bits::eTransferSrc | vk::বাফার_Usage_Flag_Bits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    যদি (device->বাফার_device_address) {
        usage_flags |= vk::বাফার_Usage_Flag_Bits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::বাফার_Create_Info বাফার_create_info{
        vk::বাফার_Create_Flags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::External_Memory_বাফার_Create_Info external_memory_bci;
    যদি (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        বাফার_create_info.setPNext(&external_memory_bci);
    }

    buf->বাফার = device->device.create_বাফার(বাফার_create_info);

    vk::MemoryRequirements mem_req = device->device.get_বাফার_Memory_Requirements(buf->বাফার);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    ধ্রুবক vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    যদি (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    যদি (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        চেষ্টা {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } ধরো (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_বাফার(buf->বাফার);
            ফেরত {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        জন্য (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            যদি (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                চালু;
            }
            যদি (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                চালু;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            যদি ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                বিরতি;
            }
        }
        যদি (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_বাফার(buf->বাফার);
            ফেরত {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        চেষ্টা {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } ধরো (ধ্রুবক vk::SystemError& e) {
        }
    } অন্যথা {
        জন্য (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ধ্রুবক auto & req_flags = *it;

            ধ্রুবক std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            যদি (memory_type_indices.empty()) {
                চালু;
            }

            যৌক্তিক done = মিথ্যা;

            জন্য (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                চেষ্টা {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = সত্য;
                    বিরতি;
                } ধরো (ধ্রুবক vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    যদি (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroy_বাফার(buf->বাফার);
                        ছোড়ো e;
                    }
                }
            }

            যদি (done) {
                বিরতি;
            }
        }
    }

    যদি (!buf->device_memory) {
        device->device.destroy_বাফার(buf->বাফার);
        ছোড়ো vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    যদি (import_ptr) {
        buf->ptr = import_ptr;
    } অন্যথা {
        যদি (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_আকার);
        }
    }

    device->device.bind_বাফার_Memory(buf->বাফার, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    যদি (device->বাফার_device_address) {
        ধ্রুবক vk::বাফার_Device_Address_Info addressInfo(buf->বাফার);
        buf->bda_addr = device->device.get_বাফার_Address(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    ফেরত buf;
}

vk_বাফার ggml_vk_create_বাফার_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    চেষ্টা {
        ফেরত ggml_vk_create_বাফার(device, size, {req_flags, fallback_flags});
    } ধরো (ধ্রুবক vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        ছোড়ো e;
    }
}

vk_বাফার ggml_vk_create_বাফার_device(vk_device& device, size_t size) {
    vk_বাফার buf;
    চেষ্টা {
        যদি (device->prefer_host_memory) {
            buf = ggml_vk_create_বাফার(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } অন্যথা যদি (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_বাফার(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } অন্যথা যদি (device->disable_host_visible_vidmem) {
            যদি (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_বাফার(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } অন্যথা {
                buf = ggml_vk_create_বাফার(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } অন্যথা {
            // use rebar if available, otherwise fallback to device only visible memory
            যদি (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_বাফার(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } অন্যথা {
                buf = ggml_vk_create_বাফার(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } ধরো (ধ্রুবক vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        ছোড়ো e;
    }

    ফেরত buf;
}

শূন্য ggml_vk_destroy_বাফার(vk_বাফার& buf) {
    যদি (buf == nullptr) {
        ফেরত;
    }

    যদি (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
স্থির size_t ggml_vk_host_allocation_find(ধ্রুবক শূন্য * ptr) {
    জন্য (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ধ্রুবক uint8_t * base = (ধ্রুবক uint8_t *) vk_instance.host_allocations[i].base;
        যদি (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            ফেরত i;
        }
    }
    ফেরত SIZE_MAX;
}

শূন্য * ggml_vk_host_malloc(size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    যদি (vk_instance.multi_device) {
        ধ্রুবক size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ধ্রুবক size_t alloc_আকার = (size + align - 1) / align * align;

        শূন্য * ptr = portable_aligned_malloc(align, alloc_আকার);
        যদি (ptr == nullptr) {
            ফেরত nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_আকার;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        ফেরত ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_বাফার buf;
    চেষ্টা {
        buf = ggml_vk_create_বাফার(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } ধরো (vk::SystemError&) {
        ফেরত nullptr;
    }

    যদি (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_বাফার(buf);
        ফেরত nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    ফেরত buf->ptr;
}

শূন্য ggml_vk_host_free(শূন্য * ptr) {
    যদি (ptr == nullptr) {
        ফেরত;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ধ্রুবক size_t সূচক = ggml_vk_host_allocation_find(ptr);
        যদি (সূচক == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            ফেরত;
        }
        entry = std::move(vk_instance.host_allocations[সূচক]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + সূচক);
    }

    জন্য (auto & view : entry.views) {
        ggml_vk_destroy_বাফার(view);
    }
    যদি (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

শূন্য ggml_vk_host_get(ধ্রুবক vk_device& device, ধ্রুবক শূন্য * ptr, vk_বাফার& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ধ্রুবক size_t সূচক = ggml_vk_host_allocation_find(ptr);
    যদি (সূচক == SIZE_MAX) {
        ফেরত;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[সূচক];

    যদি (entry.views[device->idx] == nullptr) {
        ধ্রুবক uint32_t bit = 1u << device->idx;
        যদি (!vk_instance.multi_device || !vk_instance.device_host_বাফার_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            ফেরত;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_বাফার_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        যদি (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            ফেরত;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (ধ্রুবক uint8_t *) ptr - (ধ্রুবক uint8_t *) entry.base;
}

শূন্য ggml_vk_ensure_sync_staging_বাফার(vk_device& device, size_t size) {
    যদি (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_বাফার(device->sync_staging);
        device->sync_staging = ggml_vk_create_বাফার_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

শূন্য ggml_vk_ensure_sync_staging_বাফার(ggml_backend_vk_context * ctx, size_t size) {
    যদি (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_বাফার(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_বাফার_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

স্থির শূন্য ggml_vk_বাফার_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_বাফার& dst, size_t offset, ধ্রুবক ggml_tensor * tensor, যৌক্তিক sync_staging = মিথ্যা) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    যদি(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_বাফার buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->তথ্য, buf, buf_offset);

    ধ্রুবক uint64_t ne0 = tensor->ne[0];
    ধ্রুবক uint64_t ne1 = tensor->ne[1];
    ধ্রুবক uint64_t ne2 = tensor->ne[2];
    ধ্রুবক uint64_t ne3 = tensor->ne[3];
    ধ্রুবক uint64_t nb0 = tensor->nb[0];
    ধ্রুবক uint64_t nb1 = tensor->nb[1];
    ধ্রুবক uint64_t nb2 = tensor->nb[2];
    ধ্রুবক uint64_t nb3 = tensor->nb[3];
    ধ্রুবক ggml_type type = tensor->type;
    ধ্রুবক uint64_t ts = ggml_type_আকার(type);
    ধ্রুবক uint64_t bs = ggml_blck_আকার(type);

    ধ্রুবক uint64_t dstnb0 = ts;
    ধ্রুবক uint64_t dstnb1 = dstnb0*(ne0/bs);
    ধ্রুবক uint64_t dstnb2 = dstnb1*ne1;
    ধ্রুবক uint64_t dstnb3 = dstnb2*ne2;

    ধ্রুবক uint64_t ne = ggml_nelements(tensor);

    যদি (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::বাফার_Copy> slices;

        জন্য (uint64_t i3 = 0; i3 < ne3; i3++) {
            জন্য (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                যদি (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } অন্যথা {
                    জন্য (uint64_t i1 = 0; i1 < ne1; i1++) {
                        যদি (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } অন্যথা {
                            ধ্রুবক uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            ধ্রুবক uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            জন্য (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->বাফার->buf.copy_বাফার(buf->বাফার, dst->বাফার, slices);
        ফেরত;
    }

    যদি (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_বাফার& staging = ctx->device->sync_staging;
    ধ্রুবক uint64_t copy_আকার = ts*ne/bs;
    ggml_vk_ensure_sync_staging_বাফার(ctx->device, copy_আকার);
    Vk_বাফার_Copy buf_copy{ 0, offset, copy_আকার };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_বাফার(subctx->s->বাফার->buf, (Vk_বাফার)staging->বাফার, (Vk_বাফার)dst->বাফার, 1, &buf_copy);

    জন্য (uint64_t i3 = 0; i3 < ne3; i3++) {
        জন্য (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            যদি (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ধ্রুবক uint8_t *) tensor->তথ্য + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } অন্যথা {
                জন্য (uint64_t i1 = 0; i1 < ne1; i1++) {
                    যদি (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ধ্রুবক uint8_t *) tensor->তথ্য + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } অন্যথা {
                        ধ্রুবক uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        ধ্রুবক uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        জন্য (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ধ্রুবক uint8_t *) tensor->তথ্য + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

যৌক্তিক ggml_vk_বাফার_write_2d_async(vk_context subctx, vk_বাফার& dst, size_t offset, ধ্রুবক শূন্য * src, size_t spitch, size_t dpitch, size_t width, size_t height, যৌক্তিক sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_বাফার buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    যদি (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::বাফার_Copy> slices(1);
        যদি (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } অন্যথা {
            slices.resize(height);
            জন্য (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->বাফার->buf.copy_বাফার(buf->বাফার, dst->বাফার, slices);
        ফেরত সত্য;
    }
    VK_LOG_DEBUG("STAGING");

    যদি (!sync_staging) {
        // copy was not handled caller needs to fall back
        ফেরত মিথ্যা;
    }

    // Staging buffer required
    ধ্রুবক size_t staging_আকার = width * height;
    ggml_vk_ensure_sync_staging_বাফার(dst->device, staging_আকার);

    vk_বাফার& staging_বাফার = dst->device->sync_staging;

    std::vector<vk::বাফার_Copy> slices(1);
    যদি (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_আকার;
    } অন্যথা {
        slices.resize(height);
        জন্য (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->বাফার->buf.copy_বাফার(staging_বাফার->বাফার, dst->বাফার, slices);

    যদি (width == spitch) {
        deferred_memcpy((uint8_t *)staging_বাফার->ptr, src, staging_আকার, &subctx->in_memcpys);
    } অন্যথা {
        জন্য (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_বাফার->ptr + i * width, (ধ্রুবক uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    ফেরত সত্য;
}

যৌক্তিক ggml_vk_বাফার_write_async(vk_context subctx, vk_বাফার& dst, size_t offset, ধ্রুবক শূন্য * src, size_t size, যৌক্তিক sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    ফেরত ggml_vk_বাফার_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

শূন্য ggml_vk_বাফার_write_2d(vk_বাফার& dst, size_t offset, ধ্রুবক শূন্য * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    যদি(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        যদি (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } অন্যথা {
            জন্য (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (ধ্রুবক uint8_t *) src + i * spitch, width);
            }
        }
    } অন্যথা {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        যৌক্তিক ret = ggml_vk_বাফার_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, সত্য);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        জন্য (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        জন্য (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, সত্য, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

শূন্য ggml_vk_বাফার_write(vk_বাফার& dst, size_t offset, ধ্রুবক শূন্য * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_বাফার_write_2d(dst, offset, src, size, size, size, 1);
}

যৌক্তিক ggml_vk_বাফার_read_2d_async(vk_context subctx, vk_বাফার& src, size_t offset, শূন্য * dst, size_t spitch, size_t dpitch, size_t width, size_t height, যৌক্তিক sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_বাফার buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::বাফার_Copy> slices(1);
    যদি (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } অন্যথা {
        slices.resize(height);
        জন্য (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    যদি (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->বাফার->buf.copy_বাফার(src->বাফার, buf->বাফার, slices);

        ফেরত সত্য;
    }
    VK_LOG_DEBUG("STAGING");

    যদি (!sync_staging) {
        // copy was not handled caller needs to fall back
        ফেরত মিথ্যা;
    }

    // Fall back to staging buffer
    ধ্রুবক size_t staging_আকার = width * height;
    ggml_vk_ensure_sync_staging_বাফার(src->device, staging_আকার);

    vk_বাফার& staging_বাফার = src->device->sync_staging;

    std::vector<vk::বাফার_Copy> staging_slices(1);
    যদি (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_আকার;
    } অন্যথা {
        staging_slices.resize(height);
        জন্য (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->বাফার->buf.copy_বাফার(src->বাফার, staging_বাফার->বাফার, staging_slices);

    যদি (width == dpitch) {
        deferred_memcpy(dst, staging_বাফার->ptr, staging_আকার, &subctx->out_memcpys);
    } অন্যথা {
        জন্য (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ধ্রুবক uint8_t *) staging_বাফার->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_vk_বাফার_read_async(vk_context subctx, vk_বাফার& src, size_t offset, শূন্য * dst, size_t size, যৌক্তিক sync_staging = মিথ্যা) {
    ফেরত ggml_vk_বাফার_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

শূন্য ggml_vk_বাফার_read_2d(vk_বাফার& src, size_t offset, শূন্য * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->বাফার << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ধ্রুবক যৌক্তিক slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    যদি(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->বাফার->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, সত্য, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        যদি (width == spitch && width == dpitch) {
            memcpy(dst, (ধ্রুবক uint8_t *) src->ptr + offset, width * height);
        } অন্যথা {
            জন্য (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (ধ্রুবক uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } অন্যথা {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        যৌক্তিক ret = ggml_vk_বাফার_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, সত্য);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, সত্য, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        জন্য (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

শূন্য ggml_vk_বাফার_read(vk_বাফার& src, size_t offset, শূন্য * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->বাফার << ", " << offset << ", " << size << ")");
    ggml_vk_বাফার_read_2d(src, offset, dst, size, size, size, 1);
}

শূন্য ggml_vk_বাফার_copy_async(vk_context& ctx, vk_বাফার& dst, size_t dst_offset, vk_বাফার& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_বাফার_Copy bc{ src_offset, dst_offset, size };

    vk_Cmd_Copy_বাফার(ctx->s->বাফার->buf, (Vk_বাফার)src->বাফার, (Vk_বাফার)dst->বাফার, 1, &bc);
}

শূন্য ggml_vk_বাফার_copy(vk_বাফার& dst, size_t dst_offset, vk_বাফার& src, size_t src_offset, size_t size) {
    যদি (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_বাফার_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, সত্য, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } অন্যথা {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_বাফার(src->device, size);

        // Copy to src staging buffer
        ggml_vk_বাফার_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_বাফার_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

শূন্য ggml_vk_বাফার_memset_async(vk_context& ctx, vk_বাফার& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    যদি (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        ফেরত;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->বাফার->buf.fill_বাফার(dst->বাফার, offset, size, c);
}

শূন্য ggml_vk_বাফার_memset(vk_বাফার& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    যদি (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + offset, c, size);
        ফেরত;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->বাফার->buf.fill_বাফার(dst->বাফার, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, সত্য, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_বাফার_i ggml_backend_vk_বাফার_interface = {
    /* .free_buffer     = */ ggml_backend_vk_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_vk_বাফার_get_base,
    /* .init_tensor     = */ ggml_backend_vk_বাফার_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_বাফার_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_বাফার_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_বাফার_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_বাফার_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_বাফার_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_বাফার_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};

vk_বাফার ggml_vk_বাফার_from_host_ptr(vk_device & device, শূন্য * ptr, size_t size) {
    যদি (!device->external_memory_host) {
        ফেরত {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    যদি (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        ফেরত {};
    }
    যদি (size & (device->min_imported_host_pointer_alignment - 1)) {
        ফেরত {};
    }

    ধ্রুবক vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_বাফার buf {};
    চেষ্টা {
        buf = ggml_vk_create_বাফার(device, size, { property_flags }, ptr);
    } ধরো (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    ফেরত buf;
}

