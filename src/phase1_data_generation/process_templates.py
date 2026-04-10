"""
20 unique IT process graph templates (G3-G22), each with distinct topology.

G1 and G2 come from CSV files (graph_reader.py). These 20 templates provide
genuinely different workflows covering CI/CD, security, infrastructure,
data engineering, agile, audit, and more.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .graph_reader import ProcessGraph, NodeInfo, EdgeInfo

_HUMAN_POOL = [
    "Nguyễn Văn A", "Trần Thị B", "Lê Văn C", "Phạm Thị D",
    "Hoàng Văn E", "Đặng Thị F", "Vũ Văn G", "Bùi Thị H",
    "Đỗ Văn I", "Ngô Thị K", "Dương Văn L", "Lý Thị M",
    "Phan Văn N", "Trịnh Thị O", "Đinh Văn P", "Hồ Thị Q",
    "Mai Văn R", "Tạ Thị S", "Cao Văn T", "Lưu Thị U",
]

_ROLE_POOL = [
    "PM", "Techlead", "Developer", "Senior Developer", "Tester",
    "QA Lead", "DevOps", "SRE", "DBA", "BA", "Security Engineer",
    "Network Engineer", "Cloud Engineer", "Data Engineer",
    "Scrum Master", "System Admin", "Help Desk",
]

NodeSpec = Tuple[str, str, str, float, int]
EdgeSpec = Tuple[str, str]


def _build(
    graph_id: str,
    nodes_spec: List[NodeSpec],
    edges_spec: List[EdgeSpec],
    rng: random.Random,
) -> ProcessGraph:
    pg = ProcessGraph(graph_id=graph_id)
    for nid, label, ntype, cost, hres in nodes_spec:
        humans: List[Dict[str, str]] = []
        if ntype == "Task" and hres > 0:
            for _ in range(hres):
                humans.append({
                    "human_id": f"H{rng.randint(1, 100)}",
                    "name": rng.choice(_HUMAN_POOL),
                    "role": rng.choice(_ROLE_POOL),
                })
        pg.nodes[nid] = NodeInfo(
            node_id=nid, label=label, node_type=ntype,
            graph=graph_id, cost=cost, human_res=hres, humans=humans,
        )
    for src, tgt in edges_spec:
        pg.edges.append(EdgeInfo(source=src, target=tgt, graph=graph_id))
    return pg


# ─────────────────────────────────────────────────────────────────────
# Each entry: (graph_id, nodes_list, edges_list)
# Constraints enforced by EventLogSimulator:
#   - Task / StartEvent: exactly 1 outgoing edge
#   - ExclusiveGateway: 2+ outgoing edges (branching / loop)
#   - EndEvent: 0 outgoing edges
#   - Multiple incoming edges (convergence) are OK
# ─────────────────────────────────────────────────────────────────────

_TEMPLATES: List[Tuple[str, List[NodeSpec], List[EdgeSpec]]] = [

    # ── G3: CI/CD Pipeline ───────────────────────────────────────────
    ("G3", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Commit_Code",           "Task",             30,   1),
        ("N3",  "Build_Project",         "Task",             50,   1),
        ("N4",  "Chay_Unit_Test",        "Task",             80,   1),
        ("GW1", "Kiem_Tra_Unit_Test",    "ExclusiveGateway",  0,   0),
        ("N5",  "Chay_Integration_Test", "Task",            120,   2),
        ("N6",  "Deploy_Staging",        "Task",            100,   1),
        ("N7",  "Chay_Smoke_Test",       "Task",             60,   1),
        ("GW2", "Kiem_Tra_Smoke_Test",   "ExclusiveGateway",  0,   0),
        ("N8",  "Deploy_Production",     "Task",            200,   2),
        ("N9",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "GW1"),
        ("GW1", "N5"), ("GW1", "N2"),
        ("N5", "N6"), ("N6", "N7"), ("N7", "GW2"),
        ("GW2", "N8"), ("GW2", "N6"),
        ("N8", "N9"),
    ]),

    # ── G4: Xử Lý Sự Cố Mạng ───────────────────────────────────────
    ("G4", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Nhan_Canh_Bao_Mang",   "Task",             20,   1),
        ("N3",  "Kiem_Tra_Ket_Noi",     "Task",             40,   1),
        ("N4",  "Phan_Tich_Log_Mang",   "Task",             60,   1),
        ("GW1", "Phan_Loai_Su_Co",      "ExclusiveGateway",  0,   0),
        ("N5",  "Thay_The_Thiet_Bi",    "Task",            300,   2),
        ("N6",  "Cap_Nhat_Cau_Hinh",    "Task",             80,   1),
        ("N7",  "Xac_Nhan_Ket_Noi",     "Task",             30,   1),
        ("N8",  "Bao_Cao_Su_Co_Mang",   "Task",             40,   1),
        ("N9",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "GW1"),
        ("GW1", "N5"), ("GW1", "N6"),
        ("N5", "N7"), ("N6", "N7"),
        ("N7", "N8"), ("N8", "N9"),
    ]),

    # ── G5: Quản Lý Quyền Truy Cập ──────────────────────────────────
    ("G5", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Nhan_Yeu_Cau_Cap_Quyen", "Task",           20,   1),
        ("N3",  "Xac_Minh_Danh_Tinh",   "Task",             40,   1),
        ("GW1", "Ket_Qua_Xac_Minh",     "ExclusiveGateway",  0,   0),
        ("N4",  "Tu_Choi_Yeu_Cau",      "Task",             10,   1),
        ("N5",  "Kiem_Tra_Chinh_Sach",   "Task",             60,   2),
        ("N6",  "Phe_Duyet_Quyen",      "Task",             30,   1),
        ("N7",  "Cau_Hinh_Quyen",       "Task",             50,   1),
        ("N8",  "Gui_Thong_Bao_Quyen",  "Task",             10,   1),
        ("N9",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "GW1"),
        ("GW1", "N4"), ("GW1", "N5"),
        ("N4", "N9"),
        ("N5", "N6"), ("N6", "N7"), ("N7", "N8"), ("N8", "N9"),
    ]),

    # ── G6: Quản Lý Thay Đổi (ITIL Change Management) ───────────────
    ("G6", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Dang_Ky_RFC",           "Task",             30,   1),
        ("N3",  "Danh_Gia_Rui_Ro",      "Task",             80,   2),
        ("GW1", "Muc_Do_Rui_Ro",        "ExclusiveGateway",  0,   0),
        ("N4",  "Hop_CAB",              "Task",            150,   5),
        ("N5",  "Phe_Duyet_Nhanh",      "Task",             30,   1),
        ("N6",  "Lap_Ke_Hoach_Thay_Doi", "Task",            60,   2),
        ("N7",  "Thuc_Thi_Thay_Doi",    "Task",            200,   3),
        ("N8",  "Kiem_Tra_Sau_Thay_Doi", "Task",            80,   2),
        ("GW2", "Ket_Qua_Thay_Doi",     "ExclusiveGateway",  0,   0),
        ("N9",  "Rollback_Thay_Doi",    "Task",            150,   2),
        ("N10", "Dong_Yeu_Cau_RFC",     "Task",             20,   1),
        ("N11", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "GW1"),
        ("GW1", "N4"), ("GW1", "N5"),
        ("N4", "N6"), ("N5", "N6"),
        ("N6", "N7"), ("N7", "N8"), ("N8", "GW2"),
        ("GW2", "N10"), ("GW2", "N9"),
        ("N9", "N7"),
        ("N10", "N11"),
    ]),

    # ── G7: Sao Lưu & Phục Hồi ──────────────────────────────────────
    ("G7", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Xac_Dinh_Du_Lieu",     "Task",             40,   1),
        ("N3",  "Cau_Hinh_Lich_Backup", "Task",             30,   1),
        ("N4",  "Thuc_Hien_Sao_Luu",    "Task",            100,   1),
        ("N5",  "Kiem_Tra_Toan_Ven",    "Task",             60,   1),
        ("GW1", "Ket_Qua_Toan_Ven",     "ExclusiveGateway",  0,   0),
        ("N6",  "Luu_Tru_Offsite",      "Task",             50,   1),
        ("N7",  "Cap_Nhat_Log_Backup",  "Task",             20,   1),
        ("N8",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "GW1"),
        ("GW1", "N6"), ("GW1", "N4"),
        ("N6", "N7"), ("N7", "N8"),
    ]),

    # ── G8: Ứng Phó Sự Cố Bảo Mật ──────────────────────────────────
    ("G8", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Phat_Hien_Xam_Nhap",   "Task",             30,   1),
        ("N3",  "Danh_Gia_Muc_Do",      "Task",             50,   2),
        ("GW1", "Muc_Do_Nghiem_Trong",  "ExclusiveGateway",  0,   0),
        ("N4",  "Cach_Ly_He_Thong",     "Task",            200,   3),
        ("N5",  "Ghi_Nhan_Bao_Mat",     "Task",             30,   1),
        ("N6",  "Phan_Tich_Forensic",   "Task",            300,   2),
        ("N7",  "Khac_Phuc_Lo_Hong",    "Task",            250,   2),
        ("N8",  "Khoi_Phuc_Dich_Vu",    "Task",            100,   2),
        ("N9",  "Bao_Cao_Bao_Mat",      "Task",             60,   1),
        ("N10", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "GW1"),
        ("GW1", "N4"), ("GW1", "N5"),
        ("N4", "N6"), ("N5", "N6"),
        ("N6", "N7"), ("N7", "N8"), ("N8", "N9"), ("N9", "N10"),
    ]),

    # ── G9: Mua Sắm Phần Cứng ───────────────────────────────────────
    ("G9", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Lap_Yeu_Cau_Mua",     "Task",             30,   1),
        ("N3",  "Duyet_Ngan_Sach",      "Task",             50,   2),
        ("GW1", "Ket_Qua_Duyet_NS",     "ExclusiveGateway",  0,   0),
        ("N4",  "Tu_Choi_Mua_Sam",      "Task",             10,   1),
        ("N5",  "Tim_Nha_Cung_Cap",     "Task",             60,   1),
        ("N6",  "So_Sanh_Bao_Gia",      "Task",             40,   1),
        ("N7",  "Dat_Hang_Thiet_Bi",    "Task",            500,   2),
        ("N8",  "Nhan_Hang",            "Task",             30,   1),
        ("N9",  "Kiem_Tra_Chat_Luong",  "Task",             60,   1),
        ("GW2", "Ket_Qua_KTCL",         "ExclusiveGateway",  0,   0),
        ("N10", "Nhap_Kho_CMDB",        "Task",             40,   1),
        ("N11", "Tra_Hang_NCC",         "Task",             80,   1),
        ("N12", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "GW1"),
        ("GW1", "N4"), ("GW1", "N5"),
        ("N4", "N12"),
        ("N5", "N6"), ("N6", "N7"), ("N7", "N8"), ("N8", "N9"), ("N9", "GW2"),
        ("GW2", "N10"), ("GW2", "N11"),
        ("N10", "N12"), ("N11", "N12"),
    ]),

    # ── G10: Phát Hành Phần Mềm (Release Management) ────────────────
    ("G10", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Dong_Bang_Feature",    "Task",             40,   1),
        ("N3",  "Merge_Code",           "Task",             30,   1),
        ("N4",  "Build_Release",        "Task",             60,   1),
        ("N5",  "Kiem_Tra_Regression",  "Task",            120,   3),
        ("GW1", "Ket_Qua_Regression",   "ExclusiveGateway",  0,   0),
        ("N6",  "Hotfix_Release",       "Task",            100,   2),
        ("N7",  "Deploy_UAT",           "Task",             80,   1),
        ("N8",  "Nghiem_Thu_UAT",       "Task",            150,   3),
        ("GW2", "Ket_Qua_UAT",          "ExclusiveGateway",  0,   0),
        ("N9",  "Deploy_Production_RM", "Task",            200,   2),
        ("N10", "Theo_Doi_Post_Release", "Task",            60,   2),
        ("N11", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "GW1"),
        ("GW1", "N7"), ("GW1", "N6"),
        ("N6", "N4"),
        ("N7", "N8"), ("N8", "GW2"),
        ("GW2", "N9"), ("GW2", "N6"),
        ("N9", "N10"), ("N10", "N11"),
    ]),

    # ── G11: Theo Dõi & Sửa Lỗi (Bug Tracking) ─────────────────────
    ("G11", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Bao_Cao_Loi",          "Task",             20,   1),
        ("N3",  "Phan_Loai_Loi",        "Task",             30,   1),
        ("N4",  "Phan_Tich_Loi",        "Task",             60,   1),
        ("N5",  "Sua_Loi",              "Task",            150,   2),
        ("N6",  "Code_Review_Bug",      "Task",             50,   1),
        ("GW1", "Ket_Qua_Review_Bug",   "ExclusiveGateway",  0,   0),
        ("N7",  "QA_Xac_Nhan",          "Task",             80,   2),
        ("GW2", "Ket_Qua_QA",           "ExclusiveGateway",  0,   0),
        ("N8",  "Dong_Loi",             "Task",             10,   1),
        ("N9",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"),
        ("N5", "N6"), ("N6", "GW1"),
        ("GW1", "N7"), ("GW1", "N5"),
        ("N7", "GW2"),
        ("GW2", "N8"), ("GW2", "N5"),
        ("N8", "N9"),
    ]),

    # ── G12: Di Chuyển Cơ Sở Dữ Liệu ───────────────────────────────
    ("G12", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Phan_Tich_Schema",     "Task",             80,   2),
        ("N3",  "Thiet_Ke_Migration",   "Task",            100,   2),
        ("N4",  "Viet_Script_Migration", "Task",           120,   1),
        ("N5",  "Chay_Test_Dev",        "Task",             60,   1),
        ("GW1", "Ket_Qua_Test_Dev",     "ExclusiveGateway",  0,   0),
        ("N6",  "Backup_DB_Prod",       "Task",             80,   1),
        ("N7",  "Thuc_Thi_Migration",   "Task",            200,   2),
        ("N8",  "Kiem_Tra_Du_Lieu",     "Task",            100,   2),
        ("GW2", "Ket_Qua_Du_Lieu",      "ExclusiveGateway",  0,   0),
        ("N9",  "Rollback_DB",          "Task",            150,   1),
        ("N10", "Cap_Nhat_Ung_Dung",    "Task",             60,   1),
        ("N11", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "GW1"),
        ("GW1", "N6"), ("GW1", "N4"),
        ("N6", "N7"), ("N7", "N8"), ("N8", "GW2"),
        ("GW2", "N10"), ("GW2", "N9"),
        ("N9", "N4"),
        ("N10", "N11"),
    ]),

    # ── G13: Cung Cấp Hạ Tầng Cloud ─────────────────────────────────
    ("G13", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Xac_Dinh_Yeu_Cau_Cloud", "Task",           40,   1),
        ("N3",  "Chon_Cloud_Provider",  "Task",             30,   1),
        ("N4",  "Thiet_Ke_Kien_Truc",  "Task",            100,   2),
        ("N5",  "Viet_IaC_Terraform",   "Task",            120,   1),
        ("N6",  "Apply_Infrastructure", "Task",             80,   1),
        ("N7",  "Cau_Hinh_Network",     "Task",             60,   1),
        ("N8",  "Cau_Hinh_Security",    "Task",             50,   1),
        ("N9",  "Deploy_Ung_Dung_Cloud", "Task",           100,   2),
        ("N10", "Health_Check_Cloud",   "Task",             30,   1),
        ("GW1", "Ket_Qua_Health_Check", "ExclusiveGateway",  0,   0),
        ("N11", "Giam_Sat_Chi_Phi",     "Task",             40,   1),
        ("N12", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"),
        ("N5", "N6"), ("N6", "N7"), ("N7", "N8"), ("N8", "N9"),
        ("N9", "N10"), ("N10", "GW1"),
        ("GW1", "N11"), ("GW1", "N6"),
        ("N11", "N12"),
    ]),

    # ── G14: Kiểm Tra Xâm Nhập (Penetration Testing) ────────────────
    ("G14", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Lap_Ke_Hoach_Pentest", "Task",             60,   2),
        ("N3",  "Thu_Thap_Thong_Tin",   "Task",             80,   1),
        ("N4",  "Scan_Lo_Hong",         "Task",            100,   1),
        ("N5",  "Khai_Thac_Lo_Hong",    "Task",            200,   2),
        ("GW1", "Tim_Thay_Lo_Hong",     "ExclusiveGateway",  0,   0),
        ("N6",  "Leo_Thang_Quyen",      "Task",            250,   1),
        ("N7",  "Ghi_Nhan_Ket_Qua",    "Task",             30,   1),
        ("N8",  "Viet_Bao_Cao_Pentest", "Task",             80,   1),
        ("N9",  "Trinh_Bay_Ket_Qua",   "Task",             40,   2),
        ("N10", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "GW1"),
        ("GW1", "N6"), ("GW1", "N7"),
        ("N6", "N7"),
        ("N7", "N8"), ("N8", "N9"), ("N9", "N10"),
    ]),

    # ── G15: Onboard Nhân Viên IT ────────────────────────────────────
    ("G15", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Tao_Tai_Khoan_Email",  "Task",             20,   1),
        ("N3",  "Cap_Quyen_He_Thong",   "Task",             30,   1),
        ("N4",  "Cai_Dat_May_Tinh",     "Task",             60,   1),
        ("N5",  "Cau_Hinh_VPN",         "Task",             30,   1),
        ("N6",  "Huong_Dan_Bao_Mat",    "Task",             40,   1),
        ("N7",  "Dao_Tao_Cong_Cu",      "Task",             80,   2),
        ("GW1", "Ket_Qua_Dao_Tao",      "ExclusiveGateway",  0,   0),
        ("N8",  "Cap_The_Nhan_Vien",    "Task",             20,   1),
        ("N9",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"),
        ("N5", "N6"), ("N6", "N7"), ("N7", "GW1"),
        ("GW1", "N8"), ("GW1", "N7"),
        ("N8", "N9"),
    ]),

    # ── G16: Khôi Phục Thảm Họa (Disaster Recovery) ─────────────────
    ("G16", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Phat_Hien_Su_Co_DR",   "Task",             20,   1),
        ("N3",  "Danh_Gia_Thiet_Hai",   "Task",             50,   2),
        ("GW1", "Muc_Do_Thiet_Hai",     "ExclusiveGateway",  0,   0),
        ("N4",  "Khoi_Phuc_Tung_Phan",  "Task",            200,   3),
        ("N5",  "Kich_Hoat_DR_Site",    "Task",            500,   5),
        ("N6",  "Khoi_Phuc_Du_Lieu",    "Task",            150,   2),
        ("N7",  "Kiem_Tra_He_Thong_DR", "Task",             80,   2),
        ("GW2", "Ket_Qua_KT_DR",        "ExclusiveGateway",  0,   0),
        ("N8",  "Thong_Bao_Nguoi_Dung", "Task",             20,   1),
        ("N9",  "Bao_Cao_Sau_Su_Co",    "Task",             60,   1),
        ("N10", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "GW1"),
        ("GW1", "N4"), ("GW1", "N5"),
        ("N4", "N6"), ("N5", "N6"),
        ("N6", "N7"), ("N7", "GW2"),
        ("GW2", "N8"), ("GW2", "N6"),
        ("N8", "N9"), ("N9", "N10"),
    ]),

    # ── G17: Tích Hợp API ───────────────────────────────────────────
    ("G17", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Phan_Tich_API_Spec",   "Task",             60,   1),
        ("N3",  "Thiet_Ke_Mapping",     "Task",             40,   1),
        ("N4",  "Viet_Code_Tich_Hop",   "Task",            120,   2),
        ("N5",  "Chay_Test_API",        "Task",             80,   1),
        ("GW1", "Ket_Qua_Test_API",     "ExclusiveGateway",  0,   0),
        ("N6",  "Viet_Tai_Lieu_API",    "Task",             40,   1),
        ("N7",  "Deploy_API_Gateway",   "Task",             60,   1),
        ("N8",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "GW1"),
        ("GW1", "N6"), ("GW1", "N4"),
        ("N6", "N7"), ("N7", "N8"),
    ]),

    # ── G18: Tối Ưu Hiệu Năng ──────────────────────────────────────
    ("G18", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Thiet_Lap_Monitoring", "Task",             40,   1),
        ("N3",  "Thu_Thap_Metrics",     "Task",             30,   1),
        ("N4",  "Phan_Tich_Bottleneck", "Task",             80,   2),
        ("GW1", "Tim_Thay_Bottleneck",  "ExclusiveGateway",  0,   0),
        ("N5",  "Tang_Cuong_Monitoring", "Task",            30,   1),
        ("N6",  "De_Xuat_Toi_Uu",      "Task",             60,   1),
        ("N7",  "Thuc_Hien_Toi_Uu",    "Task",            150,   2),
        ("N8",  "Load_Testing",         "Task",            100,   1),
        ("GW2", "Ket_Qua_Load_Test",    "ExclusiveGateway",  0,   0),
        ("N9",  "Cap_Nhat_Baseline",    "Task",             30,   1),
        ("N10", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "GW1"),
        ("GW1", "N6"), ("GW1", "N5"),
        ("N5", "N10"),
        ("N6", "N7"), ("N7", "N8"), ("N8", "GW2"),
        ("GW2", "N9"), ("GW2", "N4"),
        ("N9", "N10"),
    ]),

    # ── G19: Quy Trình Sprint Agile ─────────────────────────────────
    ("G19", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Sprint_Planning",      "Task",             60,   4),
        ("N3",  "Tao_User_Story",       "Task",             40,   2),
        ("N4",  "Daily_Standup",        "Task",             15,   4),
        ("N5",  "Phat_Trien_Feature",   "Task",            200,   3),
        ("N6",  "Code_Review_Sprint",   "Task",             50,   2),
        ("GW1", "Ket_Qua_Review_Sprint", "ExclusiveGateway", 0,   0),
        ("N7",  "Sprint_Review",        "Task",             40,   4),
        ("N8",  "Sprint_Retrospective", "Task",             30,   4),
        ("N9",  "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"),
        ("N5", "N6"), ("N6", "GW1"),
        ("GW1", "N7"), ("GW1", "N5"),
        ("N7", "N8"), ("N8", "N9"),
    ]),

    # ── G20: Kiểm Toán IT (IT Audit & Compliance) ───────────────────
    ("G20", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Xac_Dinh_Pham_Vi",    "Task",             40,   2),
        ("N3",  "Thu_Thap_Bang_Chung",  "Task",             80,   2),
        ("N4",  "Phong_Van_Nhan_Vien",  "Task",             60,   1),
        ("N5",  "Kiem_Tra_Chinh_Sach_IT", "Task",           50,   1),
        ("GW1", "Ket_Qua_Kiem_Tra_IT",  "ExclusiveGateway",  0,   0),
        ("N6",  "Ghi_Nhan_Vi_Pham",     "Task",             40,   1),
        ("N7",  "Ghi_Nhan_Tuan_Thu",    "Task",             20,   1),
        ("N8",  "Viet_Bao_Cao_Audit",   "Task",             80,   1),
        ("N9",  "Trinh_Bay_BLD",        "Task",             40,   2),
        ("N10", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "GW1"),
        ("GW1", "N6"), ("GW1", "N7"),
        ("N6", "N8"), ("N7", "N8"),
        ("N8", "N9"), ("N9", "N10"),
    ]),

    # ── G21: Phát Triển Microservices ────────────────────────────────
    ("G21", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Thiet_Ke_Service",     "Task",             80,   2),
        ("N3",  "Dinh_Nghia_API_Contract", "Task",          40,   1),
        ("N4",  "Viet_Code_Service",    "Task",            200,   2),
        ("N5",  "Viet_Unit_Test_MS",    "Task",             80,   1),
        ("GW1", "Ket_Qua_UT_MS",        "ExclusiveGateway",  0,   0),
        ("N6",  "Dong_Goi_Docker",      "Task",             40,   1),
        ("N7",  "Push_Registry",        "Task",             20,   1),
        ("N8",  "Deploy_K8s",           "Task",            100,   2),
        ("N9",  "Kiem_Tra_Service_Mesh", "Task",            60,   1),
        ("GW2", "Ket_Qua_Service_Mesh", "ExclusiveGateway",  0,   0),
        ("N10", "Cap_Nhat_API_Gateway", "Task",             30,   1),
        ("N11", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"), ("N5", "GW1"),
        ("GW1", "N6"), ("GW1", "N4"),
        ("N6", "N7"), ("N7", "N8"), ("N8", "N9"), ("N9", "GW2"),
        ("GW2", "N10"), ("GW2", "N8"),
        ("N10", "N11"),
    ]),

    # ── G22: Data Pipeline (ETL) ────────────────────────────────────
    ("G22", [
        ("N1",  "Bat_Dau",               "StartEvent",       0,   0),
        ("N2",  "Xac_Dinh_Nguon_DL",   "Task",             40,   1),
        ("N3",  "Thiet_Ke_Schema_DW",   "Task",             80,   2),
        ("N4",  "Viet_ETL_Script",      "Task",            120,   1),
        ("N5",  "Extract_Du_Lieu",      "Task",             60,   1),
        ("N6",  "Transform_Du_Lieu",    "Task",            100,   1),
        ("N7",  "Validate_Du_Lieu",     "Task",             50,   1),
        ("GW1", "Ket_Qua_Validate",     "ExclusiveGateway",  0,   0),
        ("N8",  "Load_Data_Warehouse",  "Task",             80,   1),
        ("N9",  "Tao_Dashboard_BI",     "Task",             60,   1),
        ("N10", "Ket_Thuc",             "EndEvent",           0,   0),
    ], [
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"),
        ("N5", "N6"), ("N6", "N7"), ("N7", "GW1"),
        ("GW1", "N8"), ("GW1", "N4"),
        ("N8", "N9"), ("N9", "N10"),
    ]),
]


def create_diverse_templates(seed: int = 42) -> Dict[str, ProcessGraph]:
    """Build 20 unique ProcessGraph objects (G3-G22) from template definitions."""
    rng = random.Random(seed)
    result: Dict[str, ProcessGraph] = {}
    for gid, nodes_spec, edges_spec in _TEMPLATES:
        result[gid] = _build(gid, nodes_spec, edges_spec, rng)
    return result
