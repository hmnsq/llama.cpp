#ضمن "transport.h"
#ضمن "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#وإلا
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <mutex>
#ضمن <optional>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <cerrno>
#  include <chrono>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
تعريفنوع SOCKET sockfd_t;
using ssize_t = __int64;
#وإلا
تعريفنوع صحيح sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
ساكن constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#𐤀𐤌 defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ساكن constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ساكن constexpr صحيح    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ساكن constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

بنية rdma_conn {
    بنية ibv_context * ctx = nullptr;
    بنية ibv_pd * pd  = nullptr;
    بنية ibv_cq * scq = nullptr;   // send completions
    بنية ibv_cq * rcq = nullptr;   // recv completions
    بنية ibv_qp * qp  = nullptr;
    بنية ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    فراغ          * tx_buf = nullptr;
    بنية ibv_mr * tx_mr  = nullptr;

    فراغ          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    بنية ibv_mr * rx_mr  = nullptr;
    صحيح             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(صحيح i) ثابت {
        أرجع static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    منطقي post_rx(صحيح i) {
        بنية ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        بنية ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        أرجع ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        𐤀𐤌 (tx_mr) ibv_dereg_mr(tx_mr);
        𐤀𐤌 (rx_mr) ibv_dereg_mr(rx_mr);
        حرر(tx_buf);
        حرر(rx_buf);
        𐤀𐤌 (qp)  ibv_destroy_qp(qp);
        𐤀𐤌 (scq) ibv_destroy_cq(scq);
        𐤀𐤌 (rcq) ibv_destroy_cq(rcq);
        𐤀𐤌 (ch)  ibv_destroy_comp_channel(ch);
        𐤀𐤌 (pd)  ibv_dealloc_pd(pd);
        𐤀𐤌 (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
بنية rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    صحيح      gid_idx = 0;
    تعداد ibv_mtu path_mtu = IBV_MTU_1024;
};

بنية rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(حجم(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

بنية socket_t::impl {
    impl(sockfd_t fd) : use_rdma(خطأ), fd(fd) {}
    ~impl();
    منطقي send_data(ثابت فراغ * data, size_t size);
    منطقي recv_data(فراغ * data, size_t size);
    منطقي flush();
    فراغ get_caps(uint8_t * local_caps);
    فراغ update_caps(ثابت uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  وإلا
    منطقي rdma_probe();
    منطقي rdma_send(ثابت فراغ * data, size_t size);
    منطقي rdma_recv(فراغ * data, size_t size);
    منطقي tcp_peer_closed();
    منطقي rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid);
    منطقي rdma_poll(بنية ibv_cq * cq, بنية ibv_wc * wc);
    منطقي rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    منطقي     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, هذا->fd);
#ifdef _WIN32
    𐤀𐤌 (fd != INVALID_SOCKET) closesocket(هذا->fd);
#وإلا
    𐤀𐤌 (fd >= 0) close(هذا->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = حجم(addr);
    𐤀𐤌 (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        أرجع std::nullopt;
    }
    rdma_gid_t target = {};
    𐤀𐤌 (addr.ss_family == AF_INET) {
        ثابت auto * a = reinterpret_cast<ثابت sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        أرجع target;
    }
    𐤀𐤌 (addr.ss_family == AF_INET6) {
        ثابت auto * a = reinterpret_cast<ثابت sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        أرجع target;
    }
    أرجع std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

منطقي socket_t::impl::tcp_peer_closed() {
    𐤀𐤌 (fd < 0) أرجع خطأ;
#ifndef _WIN32
    بنية pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    صحيح r = poll(&pfd, 1, 0);
    أرجع r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#وإلا
    أرجع خطأ;
#endif
}

منطقي socket_t::impl::rdma_probe() {
    ثابت حرف * dev_env = std::getenv("GGML_RDMA_DEV");
    ثابت حرف * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    𐤀𐤌 (!target_gid) {
        أرجع خطأ;
    }

    ثابت uint8_t ib_port = 1;
    صحيح num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    𐤀𐤌 (!devs || num_devs == 0) أرجع خطأ;

    ibv_context * ibctx = nullptr;
    ثابت حرف * matched_dev = nullptr;
    صحيح gid_idx = gid_env ? atoi(gid_env) : -1;
    صحيح gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    لكل (صحيح d = 0; d < num_devs; d++) {
        ثابت حرف * dn = ibv_get_device_name(devs[d]);
        𐤀𐤌 (dev_env && strcmp(dev_env, dn) != 0) استمر;

        ibv_context * ctx = ibv_open_device(devs[d]);
        𐤀𐤌 (!ctx) استمر;

        ibv_port_attr pa;
        𐤀𐤌 (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); استمر; }

        صحيح found_gid = gid_idx;
        صحيح found_version = IBV_GID_TYPE_IB;
        𐤀𐤌 (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            صحيح v2_idx = -1;
            صحيح v1_idx = -1;
            لكل (صحيح i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                𐤀𐤌 (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) استمر;
                𐤀𐤌 (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) استمر;
                𐤀𐤌 (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } وإلا 𐤀𐤌 (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            𐤀𐤌 (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } وإلا 𐤀𐤌 (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } وإلا {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            𐤀𐤌 (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        𐤀𐤌 (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            اكسر;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    𐤀𐤌 (!ibctx) أرجع خطأ;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    𐤀𐤌 (!rdma->pd) أرجع خطأ;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    𐤀𐤌 (!rdma->scq || !rdma->rcq) أرجع خطأ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    𐤀𐤌 (!rdma->qp) أرجع خطأ;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    𐤀𐤌 (!rdma->tx_buf || !rdma->rx_buf) أرجع خطأ;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    𐤀𐤌 (!rdma->tx_mr || !rdma->rx_mr) أرجع خطأ;

    ibv_gid local_gid;
    𐤀𐤌 (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) أرجع خطأ;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    ثابت حرف * ver_str = "";
    𐤀𐤌 (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } وإلا 𐤀𐤌 (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    أرجع true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
منطقي socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid) {
    // RESET -> INIT
    {
        بنية ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        𐤀𐤌 (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            أرجع خطأ;
        }
    }

    لكل (صحيح i = 0; i < RDMA_RX_DEPTH; i++) {
        𐤀𐤌 (!rdma->post_rx(i)) أرجع خطأ;
    }

    // INIT -> RTR
    {
        بنية ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        𐤀𐤌 (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            أرجع خطأ;
        }
    }

    // RTR -> RTS
    {
        بنية ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        𐤀𐤌 (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            أرجع خطأ;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    أرجع true;
}

// Sleep until the completion channel has an event or the TCP peer closes.
منطقي socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    بنية pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    𐤀𐤌 (poll(pfds, 2, -1) < 0) {
        أرجع errno == EINTR;
    }
    𐤀𐤌 (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        أرجع خطأ;
    }
    𐤀𐤌 (pfds[0].revents & POLLIN) {
        بنية ibv_cq * ev_cq  = nullptr;
        فراغ          * ev_ctx = nullptr;
        𐤀𐤌 (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            أرجع خطأ;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    أرجع true;
}

منطقي socket_t::impl::rdma_poll(بنية ibv_cq * cq, بنية ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    منطقي armed = خطأ;
    لكل (uint64_t s = 1; ; s++) {
        صحيح n = ibv_poll_cq(cq, 1, wc);
        𐤀𐤌 (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            𐤀𐤌 (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            أرجع wc->status == IBV_WC_SUCCESS;
        }
        𐤀𐤌 (n < 0) أرجع خطأ;
        𐤀𐤌 (armed) {
            // armed and still empty: sleep until the next completion
            𐤀𐤌 (!rdma_wait_event()) {
                أرجع خطأ;
            }
            armed = خطأ;
            استمر;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        𐤀𐤌 (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            𐤀𐤌 (ibv_req_notify_cq(cq, 0) != 0) {
                أرجع خطأ;
            }
            armed = true;
            استمر;
        }
        𐤀𐤌 ((s & 0xFFFFF) == 0) {
            𐤀𐤌 (tcp_peer_closed()) {
                أرجع خطأ;
            }
        }
    }
}

منطقي socket_t::impl::rdma_send(ثابت فراغ * data, size_t size) {
    rdma_conn * c = rdma.get();
    ثابت uint8_t * src = (ثابت uint8_t *)data;
    size_t rem = size;
    طالما (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        بنية ibv_sge sge = {};
        بنية ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        𐤀𐤌 (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } وإلا {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        𐤀𐤌 (ibv_post_send(c->qp, &wr, &bad) != 0) أرجع خطأ;
        c->last_active = std::chrono::steady_clock::now();
        بنية ibv_wc wc;
        𐤀𐤌 (!rdma_poll(c->scq, &wc)) أرجع خطأ;

        src += chunk;
        rem -= chunk;
    }
    أرجع true;
}

منطقي socket_t::impl::rdma_recv(فراغ * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    طالما (rem > 0) {
        بنية ibv_wc wc;
        𐤀𐤌 (!rdma_poll(c->rcq, &wc)) أرجع خطأ;

        صحيح slot = (صحيح)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        𐤀𐤌 (!c->post_rx(slot)) أرجع خطأ;

        dst += got;
        rem -= got;
    }
    أرجع true;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

منطقي socket_t::impl::send_data(ثابت فراغ * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    𐤀𐤌 (use_rdma) {
        أرجع rdma->send(data, size);
    }
#وإلاإذا defined(GGML_RPC_RDMA)
    𐤀𐤌 (use_rdma) {
        أرجع rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    طالما (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (ثابت حرف *)data + bytes_sent, size_to_send, 0);
        𐤀𐤌 (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            أرجع خطأ;
        }
        bytes_sent += (size_t)n;
    }
    أرجع true;
}

منطقي socket_t::impl::recv_data(فراغ * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    𐤀𐤌 (use_rdma) {
        أرجع rdma->recv(data, size);
    }
#وإلاإذا defined(GGML_RPC_RDMA)
    𐤀𐤌 (use_rdma) {
        أرجع rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    طالما (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (حرف *)data + bytes_recv, size_to_recv, 0);
        𐤀𐤌 (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            أرجع خطأ;
        }
        𐤀𐤌 (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            أرجع خطأ;
        }
        bytes_recv += (size_t)n;
    }
    أرجع true;
}

فراغ socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    𐤀𐤌 (std::getenv("GGML_RPC_NO_RDMA")) {
        أرجع;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    𐤀𐤌 (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    𐤀𐤌 (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  وإلا
    rdma_local = {};
    𐤀𐤌 (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, حجم(rc));
    } وإلا {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

فراغ socket_t::impl::update_caps(ثابت uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    منطقي remote_rdma = خطأ;
    لكل (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    𐤀𐤌 (!rdma || !remote_rdma) {
        𐤀𐤌 (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        أرجع;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    منطقي activated = rdma->activate(remote_caps);
#  وإلا
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, حجم(rc));
    منطقي activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    𐤀𐤌 (activated) {
        use_rdma = true;
    } وإلا {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#وإلا
    (فراغ)remote_caps;
#endif // GGML_RPC_RDMA
}

منطقي socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    𐤀𐤌 (use_rdma) {
        أرجع rdma->flush();
    }
#endif
    أرجع true;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = افتراضي;

منطقي socket_t::send_data(ثابت فراغ * data, size_t size) {
    أرجع pimpl->send_data(data, size);
}

منطقي socket_t::recv_data(فراغ * data, size_t size) {
    أرجع pimpl->recv_data(data, size);
}

منطقي socket_t::flush() {
    أرجع pimpl->flush();
}

فراغ socket_t::get_caps(uint8_t * local_caps) {
    أرجع pimpl->get_caps(local_caps);
}

فراغ socket_t::update_caps(ثابت uint8_t * remote_caps) {
    أرجع pimpl->update_caps(remote_caps);
}

ثابت حرف * socket_t::transport_name() ثابت {
    أرجع pimpl->use_rdma ? "RDMA" : "TCP";
}

ساكن منطقي is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    أرجع sockfd != INVALID_SOCKET;
#وإلا
    أرجع sockfd >= 0;
#endif
}

ساكن منطقي set_no_delay(sockfd_t sockfd) {
    صحيح flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    صحيح ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (حرف *)&flag, حجم(صحيح));
    أرجع ret == 0;
}

ساكن منطقي set_reuse_addr(sockfd_t sockfd) {
    صحيح flag = 1;
    صحيح ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (حرف *)&flag, حجم(صحيح));
    أرجع ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, عدم, عدم);
    𐤀𐤌 (!is_valid_fd(client_socket_fd)) {
        أرجع nullptr;
    }
    𐤀𐤌 (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        أرجع nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ثابت حرف * host, صحيح port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    𐤀𐤌 (!is_valid_fd(sockfd)) {
        أرجع nullptr;
    }
    𐤀𐤌 (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        أرجع nullptr;
    }
    𐤀𐤌 (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        أرجع nullptr;
    }
    بنية sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    𐤀𐤌 (bind(sockfd, (بنية sockaddr *) &serv_addr, حجم(serv_addr)) < 0) {
        أرجع nullptr;
    }
    𐤀𐤌 (listen(sockfd, 1) < 0) {
        أرجع nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ثابت حرف * host, صحيح port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    𐤀𐤌 (!is_valid_fd(sockfd)) {
        أرجع nullptr;
    }
    𐤀𐤌 (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        أرجع nullptr;
    }
    بنية sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    بنية hostent * server = gethostbyname(host);
    𐤀𐤌 (server == عدم) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        أرجع nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    𐤀𐤌 (::connect(sockfd, (بنية sockaddr *)&addr, حجم(addr)) < 0) {
        أرجع nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
ساكن std::mutex g_rpc_transport_mu;
ساكن منطقي g_rpc_transport_wsa_started = خطأ;
#endif

منطقي rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    𐤀𐤌 (g_rpc_transport_wsa_started) {
        أرجع true;
    }
    WSADATA wsaData;
    صحيح res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    𐤀𐤌 (res != 0) {
        أرجع خطأ;
    }
    g_rpc_transport_wsa_started = true;
    أرجع true;
#وإلا
    أرجع true;
#endif
}

فراغ rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    𐤀𐤌 (!g_rpc_transport_wsa_started) {
        أرجع;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = خطأ;
#endif
}
