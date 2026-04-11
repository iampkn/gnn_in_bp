"""API route: /explain-net - XAI: explain a discovered Petri net using OpenAI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


class PlaceInfo(BaseModel):
    inputs: List[str]
    outputs: List[str]


class DiscoveredNet(BaseModel):
    rank: int
    log_probability: float
    num_places: int
    transitions: List[str]
    places: List[PlaceInfo]


class ExplainRequest(BaseModel):
    graph_id: str
    discovered_net: DiscoveredNet


def _load_template_graph(graph_id: str) -> Optional[dict]:
    """Load the template graph structure for context."""
    try:
        from src.phase1_data_generation.graph_reader import GraphReader
        from src.phase1_data_generation.graph_generator import GraphGenerator

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = project_root / "src" / "data" / "csv"

        reader = GraphReader(data_dir)
        base_graphs = reader.read_all()
        generator = GraphGenerator(seed=42)
        template_graphs = generator.generate_variants(base_graphs, num_variants=20)
        graphs = {**base_graphs, **template_graphs}

        if graph_id not in graphs:
            return None

        pg = graphs[graph_id]
        nodes = []
        for nid, node in pg.nodes.items():
            nodes.append({
                "id": nid,
                "label": node.label,
                "type": node.node_type,
                "cost": node.cost,
                "human_res": node.human_res,
            })

        edges = [{"source": e.source, "target": e.target} for e in pg.edges]

        return {
            "graph_id": graph_id,
            "total_cost": pg.total_cost(),
            "nodes": nodes,
            "edges": edges,
        }
    except Exception:
        return None


def _build_prompt(template: dict, net: DiscoveredNet) -> str:
    """Build a detailed prompt for OpenAI to explain the discovered net."""
    nodes_desc = []
    for n in template["nodes"]:
        label = n["label"].replace("_", " ")
        if n["type"] == "StartEvent":
            nodes_desc.append(f"  - [{n['id']}] {label} (Bắt đầu)")
        elif n["type"] == "EndEvent":
            nodes_desc.append(f"  - [{n['id']}] {label} (Kết thúc)")
        elif n["type"] == "ExclusiveGateway":
            nodes_desc.append(f"  - [{n['id']}] {label} (Nút quyết định/rẽ nhánh)")
        else:
            nodes_desc.append(
                f"  - [{n['id']}] {label} (Task, chi phí: {n['cost']}, nhân sự: {n['human_res']})"
            )

    edges_desc = []
    node_map = {n["id"]: n["label"].replace("_", " ") for n in template["nodes"]}
    for e in template["edges"]:
        src_label = node_map.get(e["source"], e["source"])
        tgt_label = node_map.get(e["target"], e["target"])
        edges_desc.append(f"  - {src_label} → {tgt_label}")

    places_desc = []
    for p in net.places:
        ins = ", ".join(t.replace("_", " ") for t in p.inputs)
        outs = ", ".join(t.replace("_", " ") for t in p.outputs)
        places_desc.append(f"  - {{{ins}}} → {{{outs}}}")

    transitions_str = ", ".join(t.replace("_", " ") for t in net.transitions)

    return f"""Bạn là chuyên gia Process Mining và tối ưu quy trình IT. Hãy phân tích và giải thích cho nhà quản lý (không có kiến thức kỹ thuật sâu) về kết quả khám phá quy trình dưới đây.

## SƠ ĐỒ QUY TRÌNH GỐC (Template Graph {template['graph_id']})

Tổng chi phí quy trình: {template['total_cost']}

Các bước trong quy trình:
{chr(10).join(nodes_desc)}

Luồng thực hiện:
{chr(10).join(edges_desc)}

## KẾT QUẢ KHÁM PHÁ TỪ MÔ HÌNH GNN — Petri Net #{net.rank}

- Xếp hạng: #{net.rank} (trong các ứng viên, #1 là tốt nhất)
- Điểm log-probability: {net.log_probability:.4f} (càng gần 0 càng khớp dữ liệu thực tế)
- Số transitions (bước): {len(net.transitions)}
- Transitions: {transitions_str}
- Số places đã chọn: {net.num_places}
- Cấu trúc Place:
{chr(10).join(places_desc) if places_desc else '  (không có place nào được chọn)'}

## YÊU CẦU PHÂN TÍCH

Hãy trả lời bằng tiếng Việt, rõ ràng và dễ hiểu cho nhà quản lý:

1. **Giải thích ý nghĩa**: Petri Net #{net.rank} này đang nói gì về quy trình? Place structure có ý nghĩa gì trong ngữ cảnh nghiệp vụ?
2. **So sánh với quy trình gốc**: Mô hình khám phá có khớp với thiết kế gốc không? Có phát hiện gì khác biệt không?
3. **Đề xuất tối ưu**: Dựa trên cấu trúc quy trình, có thể rút ngắn/tối ưu ở đâu? Bước nào có chi phí cao nhất cần chú ý? Có thể song song hóa bước nào?
4. **Rủi ro và lưu ý**: Có bước nào là điểm nghẽn (bottleneck)? Gateway nào cần giám sát chặt?

Trả lời ngắn gọn, có cấu trúc, dùng bullet points. Tập trung vào giá trị thực tiễn cho quản lý."""

    return prompt


@router.post("/explain-net")
async def explain_net(req: ExplainRequest):
    """Use OpenAI to explain a discovered Petri net in business terms."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured on the server.",
        )

    template = _load_template_graph(req.graph_id)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"Template graph '{req.graph_id}' not found.",
        )

    prompt = _build_prompt(template, req.discovered_net)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia tư vấn Process Mining cho doanh nghiệp IT. "
                        "Hãy phân tích kết quả khám phá quy trình và đưa ra lời khuyên "
                        "thiết thực, dễ hiểu cho nhà quản lý. Trả lời bằng tiếng Việt."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        explanation = response.choices[0].message.content
        return JSONResponse(content={"explanation": explanation})

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="OpenAI library not installed. Add 'openai' to requirements.txt.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
