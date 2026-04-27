import os
import re
import json
import base64
import numpy as np
import pandas as pd
from io import BytesIO
from collections import Counter
from langchain_core.messages import HumanMessage

from viser_utils import apply_viser_scaffolding, get_viser_prompt
from scaffold_utils import apply_scaffold_coordinates, get_scaffold_prompt
from dsg_utils_vasu import run_dsg_loop


def encode_image(img):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


GUIDE = """
## Answer Instruction 
Please provide an answer to the question outlined above. Your response should adhere to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:
{{
    "solution": "[Detailed step-by-step explanation]",
    "short answer": "[Concise Answer]"
}}
"""

def get_instruction_suffix(answer_type):
    # Determine instruction suffix
    if answer_type == 'multiple choice':
        return "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
    elif answer_type == 'float':
        return "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
    else:
        return "Float numbers in the answer should be formatted as three-digit floating-point numbers."


class BaselineMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        inst = get_instruction_suffix(answer_type)
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {question}\n" + GUIDE.format(INST=inst)
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )
        return message


class SushilOracleMethod:
    def __init__(self):
        self.oracle_data = pd.read_json("src/sushiloracle_data.jsonl", lines=True)

    def __call__(self, question_id, question, answer_type, subject, img, llm):
        inst = get_instruction_suffix(answer_type)

        plot_code = self.oracle_data.loc[self.oracle_data['question_id'] == int(question_id), 'plot_function'].values[0]
        plot_code_prompt = f"\n## Plot Code \nHere is the python code used to generate the image. \n```python\n{plot_code}\n```"

        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {question}\n" + GUIDE.format(INST=inst) + plot_code_prompt
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )
        return message


class SushilMethod:
    def __call__(self, question_id, question, answer_type, subject, img, llm):
        inst = get_instruction_suffix(answer_type)

        plot_code = self.generate_plot_code(question_id, img, llm)
        if plot_code:
            plot_code_prompt = f"\n## Plot Code \nHere is the python code used to generate the image. \n```python\n{plot_code}\n```"
        else:
            plot_code_prompt = ""

        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {question}\n" + GUIDE.format(INST=inst) + plot_code_prompt
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )
        return message    

    def generate_plot_code(self, question_id, img, llm):
        message = HumanMessage(
            content =[
                {
                    "type": "text",
                    "text": "Analyze this image and write a complete Python script using the matplotlib library to recreate it visually. Break the image down into basic shapes, lines, and colors. Output ONLY valid, executable Python code. Do not include any markdown formatting, explanations, or conversational text."
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"}
                }
            ]
        )

        try:
            response = llm.generate(
                [[message]], 
                n=1,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0
                },
                "chat_template_kwargs": {"enable_thinking": False}
            )
            generation_text = response.generations[0][0].text 
        except Exception as e:
            print(f"\nPlot Generation failed for ID {question_id}: {e}")
            generation_text = None
        
        return generation_text

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
class AkibMethod:
    """
    Dual-path CV augmentation pipeline.

    Option A — CV skeletonization + multi-OCR ensemble (PaddleOCR / EasyOCR /
               pytesseract). No extra LLM calls; fast.

    Option B — Same CV skeleton for structure, then uses the passed ChatOpenAI
               (qwen3.5) for batch node-label reading and edge-weight extraction.
               Includes edge attention re-ranking: each edge-midpoint crop is
               shown to the VLM with an explicit "NONE" option to suppress
               hallucinated weights.

    Both options share the CV structure pipeline. The higher-confidence result
    wins. If neither clears CONF_GATE the method falls back to the plain
    baseline prompt.
    """

    CONF_GATE = 0.45
    ALLOW_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-"

    _REASONER_PROMPT = (
        "You are answering a graph theory question. Below the question is a "
        "TEXTUAL DESCRIPTION of the graph extracted by a CV+OCR pipeline. "
        "EXTRACTION_CONFIDENCE indicates how reliable the extraction is.\n\n"
        "USAGE GUIDELINES:\n"
        "  1. If EXTRACTION_CONFIDENCE >= 0.6, use the description as a secondary "
        "reference to verify your visual reading — your visual parsing takes precedence.\n"
        "  2. If EXTRACTION_CONFIDENCE < 0.6, treat the description as unreliable "
        "background context only.\n"
        "  3. If the description and image disagree on node/edge COUNT, trust the image.\n"
        "  4. If LABEL_TYPE is 'structural_index', node names (N0, N1, …) are "
        "positional placeholders — do NOT cite them as answer values.\n"
        "  5. Never let the description override an answer you would give from the "
        "image alone unless it clearly resolves an ambiguity.\n\n"
        "{universal_block}\n\nQUESTION: {question}"
    )

    def __init__(self):
        self._paddle = None
        self._easy = None
        self._paddle_ok = False
        self._easy_ok = False

    # ── OCR engine lazy loaders (Option A) ───────────────────────────────────

    def _load_paddle(self):
        if self._paddle is not None:
            return
        try:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(use_textline_orientation=False, lang="en", device="cpu")
            self._paddle_ok = True
        except Exception:
            self._paddle_ok = False

    def _load_easy(self):
        if self._easy is not None:
            return
        try:
            import easyocr
            self._easy = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._easy_ok = True
        except Exception:
            self._easy_ok = False

    # ── Shared CV helpers ────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(img_bgr):
        import cv2
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=25, C=8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return gray, cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _skeletonize(binary):
        from skimage.morphology import skeletonize
        from skimage.util import img_as_ubyte
        return img_as_ubyte(skeletonize(binary > 0))

    @staticmethod
    def _count_neighbours(skel, y, x):
        h, w = skel.shape
        return sum(
            1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
            if (dy, dx) != (0, 0)
            and 0 <= y + dy < h and 0 <= x + dx < w
            and skel[y + dy, x + dx] > 0
        )

    def _prune_spurs(self, skel, max_spur):
        s = skel.copy()
        for _ in range(max_spur):
            ys, xs = np.where(s > 0)
            ends = [(y, x) for y, x in zip(ys, xs)
                    if self._count_neighbours(s, y, x) == 1]
            if not ends:
                break
            for y, x in ends:
                s[y, x] = 0
        return s

    @staticmethod
    def _mask_nodes(skeleton, nodes, radii, margin=6):
        import cv2
        masked = skeleton.copy()
        for i, (cx, cy) in nodes.items():
            cv2.circle(masked, (cx, cy), radii[i] + margin, 0, -1)
        return masked

    def _extract_edges(self, skel_edges, nodes):
        import cv2
        if not nodes:
            return []
        _, label_map = cv2.connectedComponents(skel_edges)
        edges = []
        for cid in range(1, label_map.max() + 1):
            comp = (label_map == cid).astype(np.uint8) * 255
            if np.count_nonzero(comp) < 20:
                continue
            ys, xs = np.where(comp > 0)
            pts = list(zip(xs.tolist(), ys.tolist()))
            ends = [(x, y) for x, y in pts if self._count_neighbours(comp, y, x) == 1]
            if len(ends) < 2:
                ends = [pts[0], pts[-1]] if len(pts) >= 2 else []
            if len(ends) < 2:
                continue
            if len(ends) > 2:
                best, t1, t2 = 0, ends[0], ends[1]
                for i in range(len(ends)):
                    for j in range(i + 1, len(ends)):
                        d = np.hypot(ends[i][0] - ends[j][0], ends[i][1] - ends[j][1])
                        if d > best:
                            best, t1, t2 = d, ends[i], ends[j]
                ends = [t1, t2]
            n1 = min(nodes, key=lambda k: np.hypot(ends[0][0] - nodes[k][0], ends[0][1] - nodes[k][1]))
            n2 = min(nodes, key=lambda k: np.hypot(ends[1][0] - nodes[k][0], ends[1][1] - nodes[k][1]))
            if n1 != n2:
                e = tuple(sorted([n1, n2]))
                if e not in edges:
                    edges.append(e)
        return edges

    def _run_cv_structure(self, img_bgr):
        """Shared first stage: detect nodes and edges via skeleton."""
        import cv2
        img_h, img_w = img_bgr.shape[:2]
        _, binary = self._preprocess(img_bgr)
        skel_raw = self._skeletonize(binary)
        img_diag = (img_h ** 2 + img_w ** 2) ** 0.5
        max_spur = max(8, int(12 * img_diag / 724))
        skeleton = self._prune_spurs(skel_raw, max_spur)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        scale = min(img_h, img_w) / 512.0
        min_r, max_r = max(8, int(10 * scale)), max(30, int(45 * scale))
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=min_r * 2, param1=50, param2=28,
                                   minRadius=min_r, maxRadius=max_r)
        raw_nodes = []
        if circles is not None:
            for cx, cy, r in circles[0].astype(int):
                if not any(np.hypot(cx - mx, cy - my) < 20 for mx, my, _ in raw_nodes):
                    raw_nodes.append((int(cx), int(cy), int(r)))
            if len(raw_nodes) > 1:
                med_r = np.median([r for _, _, r in raw_nodes])
                raw_nodes = [(cx, cy, r) for cx, cy, r in raw_nodes if r < med_r * 2.5]

        if len(raw_nodes) < 2:
            return None

        raw_nodes.sort(key=lambda t: (t[1], t[0]))
        nodes_dict = {i: (raw_nodes[i][0], raw_nodes[i][1]) for i in range(len(raw_nodes))}
        radii_dict = {i: raw_nodes[i][2] for i in range(len(raw_nodes))}

        skel_edges = self._mask_nodes(skeleton, nodes_dict, radii_dict)
        edges_idx = self._extract_edges(skel_edges, nodes_dict)
        if not edges_idx and len(nodes_dict) >= 2:
            short = max(4, max_spur // 2)
            skel2 = self._mask_nodes(self._prune_spurs(skel_raw, short), nodes_dict, radii_dict)
            edges2 = self._extract_edges(skel2, nodes_dict)
            if edges2:
                edges_idx = edges2

        return raw_nodes, nodes_dict, radii_dict, edges_idx

    # ── Option A: OCR ensemble ────────────────────────────────────────────────

    @staticmethod
    def _upscale(roi, target=80):
        import cv2
        h, w = roi.shape[:2]
        s = min(h, w)
        if s == 0:
            return roi
        scale = max(1, int(round(target / s)))
        return cv2.resize(roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _binarize(roi):
        import cv2
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw if (bw == 0).sum() < (bw == 255).sum() else 255 - bw

    def _ocr_ensemble(self, roi_bgr, mode="any"):
        import cv2
        if roi_bgr is None or roi_bgr.size == 0 or min(roi_bgr.shape[:2]) < 4:
            return "", 0.0
        roi_big = self._upscale(roi_bgr)
        bw_3ch = cv2.cvtColor(self._binarize(roi_big), cv2.COLOR_GRAY2BGR)
        allow = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" if mode == "alpha"
                 else ("0123456789.-" if mode == "num" else self.ALLOW_CHARS))
        candidates = []

        self._load_paddle()
        if self._paddle_ok:
            try:
                result = self._paddle.ocr(bw_3ch)
                if result and result[0]:
                    chunks = []
                    for item in result[0]:
                        if isinstance(item, dict):
                            t = item.get("rec_text", item.get("transcription", ""))
                            c = float(item.get("rec_score", item.get("score", 0.0)))
                        else:
                            t, c = item[1][0], float(item[1][1])
                        if t:
                            chunks.append((t, c))
                    if chunks:
                        candidates.append(("".join(ch[0] for ch in chunks).strip(),
                                           float(np.mean([ch[1] for ch in chunks])), "p"))
            except Exception:
                pass

        self._load_easy()
        if self._easy_ok:
            try:
                res = self._easy.readtext(bw_3ch, allowlist=allow, detail=1, paragraph=False)
                if res:
                    txt = "".join(t for _, t, _ in res).strip()
                    conf = float(max(c for *_, c in res))
                    if txt:
                        candidates.append((txt, conf, "e"))
            except Exception:
                pass

        if not candidates or max(c for _, c, _ in candidates) < 0.65:
            try:
                import pytesseract
                psm = "10" if mode == "alpha" else "7"
                cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist={allow}"
                gray_roi = cv2.cvtColor(roi_big, cv2.COLOR_BGR2GRAY)
                tess_txt = pytesseract.image_to_string(gray_roi, config=cfg).strip()
                data = pytesseract.image_to_data(gray_roi, config=cfg,
                                                  output_type=pytesseract.Output.DICT)
                confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c > 0]
                tess_conf = float(sum(confs) / len(confs)) / 100.0 if confs else 0.4
                if tess_txt:
                    candidates.append((tess_txt, tess_conf, "tess"))
            except Exception:
                pass

        if not candidates:
            return "", 0.0

        def _matches(s):
            s = s.strip()
            if mode == "alpha": return bool(re.fullmatch(r"[A-Za-z]+", s))
            if mode == "num":   return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s))
            return bool(s)

        pool = [(t, c, src) for t, c, src in candidates if _matches(t)] or candidates
        counts = Counter(t for t, _, _ in pool)
        top_text, top_n = counts.most_common(1)[0]
        winner_conf = max(c for t, c, _ in pool if t == top_text)
        agreement = top_n / len(pool)
        if len(pool) == 1:
            winner_conf = min(winner_conf, 0.60)
        return top_text, float(winner_conf * agreement)

    def _read_label_A(self, img_bgr, cx, cy, r):
        H, W = img_bgr.shape[:2]
        m = max(int(r * 1.05), 10)
        roi = img_bgr[max(cy - m, 0):min(cy + m, H), max(cx - m, 0):min(cx + m, W)]
        txt, conf = self._ocr_ensemble(roi, mode="alpha")
        if txt and conf >= 0.55:
            return txt, conf
        m2 = max(int(r * 1.6), 18)
        roi2 = img_bgr[max(cy - m2, 0):min(cy + m2, H), max(cx - m2, 0):min(cx + m2, W)]
        txt2, conf2 = self._ocr_ensemble(roi2, mode="any")
        return (txt2 or txt or "?"), (conf2 if txt2 else conf)

    def _read_weight_A(self, img_bgr, p1, p2):
        import cv2
        H, W = img_bgr.shape[:2]
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        if abs(angle) > 90:
            angle += 180
        side = 30
        patch = img_bgr[max(my - side, 0):min(my + side, H),
                        max(mx - side, 0):min(mx + side, W)]
        if patch.size == 0:
            return None, 0.0
        M = cv2.getRotationMatrix2D((patch.shape[1] // 2, patch.shape[0] // 2), angle, 1.0)
        rot = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]),
                              borderValue=(255, 255, 255))
        ph, pw = rot.shape[:2]
        cx_, cy_ = pw // 2, ph // 2
        hw, hh = 22, 14
        for dy in (-hh - 4, hh + 4):
            ry1, ry2 = max(cy_ + dy - hh, 0), min(cy_ + dy + hh, ph)
            roi = rot[ry1:ry2, max(cx_ - hw, 0):min(cx_ + hw, pw)]
            txt, conf = self._ocr_ensemble(roi, mode="num")
            m = re.search(r"-?\d+(?:\.\d+)?", txt or "")
            if m and conf >= 0.5:
                return float(m.group()), conf
        roi = rot[max(cy_ - hh, 0):min(cy_ + hh, ph), max(cx_ - hw, 0):min(cx_ + hw, pw)]
        txt, conf = self._ocr_ensemble(roi, mode="num")
        m = re.search(r"-?\d+(?:\.\d+)?", txt or "")
        return (float(m.group()), conf) if m else (None, 0.0)

    def _extract_optionA(self, img_bgr, raw_nodes, nodes_dict, edges_idx, question):
        """Option A: pure CV + multi-OCR ensemble."""
        node_labels, node_confs = [], []
        for cx, cy, r in raw_nodes:
            lbl, conf = self._read_label_A(img_bgr, cx, cy, r)
            node_labels.append(lbl)
            node_confs.append(conf)

        valid = [l for l in node_labels if l and l != "?"]
        if valid:
            is_alpha = [bool(re.fullmatch(r"[A-Za-z]+", l)) for l in valid]
            is_num   = [bool(re.fullmatch(r"\d+", l)) for l in valid]
            forced = ("alpha" if sum(is_alpha) / len(is_alpha) >= 0.8
                      else "num" if sum(is_num) / len(is_num) >= 0.8 else None)
            if forced:
                for i, (lbl, conf) in enumerate(zip(node_labels, node_confs)):
                    if conf < 0.55 or lbl == "?":
                        cx, cy, r = raw_nodes[i]
                        H, W = img_bgr.shape[:2]
                        m = max(int(r * 1.05), 10)
                        roi = img_bgr[max(cy - m, 0):min(cy + m, H),
                                      max(cx - m, 0):min(cx + m, W)]
                        new_lbl, new_conf = self._ocr_ensemble(roi, mode=forced)
                        if new_lbl and new_conf >= conf:
                            node_labels[i], node_confs[i] = new_lbl, new_conf

        seen = {}
        for i, lab in enumerate(node_labels):
            node_labels[i] = f"{lab}{i}" if lab in seen else lab
            seen.setdefault(lab, i)

        weights_list, weight_confs = [], []
        for u, v in edges_idx:
            w, c = self._read_weight_A(img_bgr, nodes_dict[u], nodes_dict[v])
            weights_list.append(w)
            weight_confs.append(c)

        return self._finalize(node_labels, node_confs, edges_idx, weights_list,
                              weight_confs, question, source="optionA")

    # ── Option B: VLM-based reading with edge attention re-ranking ───────────

    @staticmethod
    def _bgr_crop_b64(img_bgr, cx, cy, margin):
        import cv2
        H, W = img_bgr.shape[:2]
        crop = img_bgr[max(cy - margin, 0):min(cy + margin, H),
                       max(cx - margin, 0):min(cx + margin, W)]
        if crop.size == 0:
            return None
        _, buf = cv2.imencode(".png", crop)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _vlm_call(self, llm, content_blocks):
        """Single LangChain LLM call with thinking disabled; returns text or ''."""
        try:
            msg = HumanMessage(content=content_blocks)
            response = llm.generate(
                [[msg]],
                n=1,
                temperature=0.0,
                max_tokens=128,
                chat_template_kwargs={"enable_thinking": False},
            )
            return (response.generations[0][0].text or "").strip()
        except Exception:
            return ""

    def _vlm_read_nodes(self, img_bgr, raw_nodes, llm):
        """Send all node crops in one VLM call. Returns (labels, confs)."""
        n = len(raw_nodes)
        content = []
        valid_idx = []
        for i, (cx, cy, r) in enumerate(raw_nodes):
            b64 = self._bgr_crop_b64(img_bgr, cx, cy, max(int(r * 1.6), 18))
            if b64:
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}})
                valid_idx.append(i)

        if not valid_idx:
            return ["?"] * n, [0.0] * n

        content.append({
            "type": "text",
            "text": (
                f"These are {len(valid_idx)} node crops from a graph image (one per image, in order). "
                "For each crop state the single letter or number label inside the circle, "
                "or NONE if unclear. "
                f"Reply ONLY with a JSON array of exactly {len(valid_idx)} strings, "
                "e.g. [\"A\",\"B\",\"NONE\"]. No other text."
            )
        })

        raw = self._vlm_call(llm, content)
        try:
            parsed = json.loads(raw)
            if not (isinstance(parsed, list) and len(parsed) == len(valid_idx)):
                raise ValueError
        except Exception:
            m = re.search(r"\[.*?\]", raw, re.S)
            try:
                parsed = json.loads(m.group()) if m else []
            except Exception:
                parsed = []

        labels_map = {vi: str(v).strip() for vi, v in zip(valid_idx, parsed)}
        node_labels, node_confs = [], []
        for i in range(n):
            lbl = labels_map.get(i, "NONE")
            if lbl.upper() == "NONE" or not lbl:
                node_labels.append("?")
                node_confs.append(0.0)
            else:
                node_labels.append(lbl)
                node_confs.append(0.80)
        return node_labels, node_confs

    def _vlm_rerank_edges(self, img_bgr, edges_idx, nodes_dict, llm):
        """
        Edge attention re-ranking: send all edge-midpoint crops in one VLM
        call with an explicit NONE option to suppress hallucinated weights.
        Returns (weights_list, weight_confs).
        """
        if not edges_idx:
            return [], []

        content = []
        crop_order = []
        for u, v in edges_idx:
            cx = (nodes_dict[u][0] + nodes_dict[v][0]) // 2
            cy = (nodes_dict[u][1] + nodes_dict[v][1]) // 2
            b64 = self._bgr_crop_b64(img_bgr, cx, cy, margin=35)
            if b64:
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}})
                crop_order.append((u, v))

        if not crop_order:
            return [None] * len(edges_idx), [0.0] * len(edges_idx)

        m = len(crop_order)
        content.append({
            "type": "text",
            "text": (
                f"These are {m} edge-midpoint crops from a graph (one per image, in order). "
                "Does each crop contain a numeric weight label near its center? "
                "If yes reply with the number (e.g. '5' or '12.5'). "
                "If no number is visible reply NONE. "
                f"Reply ONLY with a JSON array of exactly {m} strings, "
                "e.g. [\"5\",\"NONE\",\"12\"]. No other text."
            )
        })

        raw = self._vlm_call(llm, content)
        try:
            parsed = json.loads(raw)
            if not (isinstance(parsed, list) and len(parsed) == m):
                raise ValueError
        except Exception:
            match = re.search(r"\[.*?\]", raw, re.S)
            try:
                parsed = json.loads(match.group()) if match else []
            except Exception:
                parsed = []

        result_map = {}
        for i, (u, v) in enumerate(crop_order):
            val = parsed[i].strip() if i < len(parsed) else "NONE"
            if val.upper() == "NONE" or not val:
                result_map[(u, v)] = (None, 0.0)
            else:
                num = re.search(r"-?\d+(?:\.\d+)?", val)
                result_map[(u, v)] = (float(num.group()), 0.75) if num else (None, 0.0)

        weights_list, weight_confs = [], []
        for u, v in edges_idx:
            w, c = result_map.get((u, v), (None, 0.0))
            weights_list.append(w)
            weight_confs.append(c)
        return weights_list, weight_confs

    def _extract_optionB(self, img_bgr, raw_nodes, nodes_dict, edges_idx, question, llm):
        """Option B: CV structure + qwen3.5 VLM for labels and edge re-ranking."""
        node_labels, node_confs = self._vlm_read_nodes(img_bgr, raw_nodes, llm)

        seen = {}
        for i, lab in enumerate(node_labels):
            node_labels[i] = f"{lab}{i}" if lab in seen else lab
            seen.setdefault(lab, i)

        weights_list, weight_confs = self._vlm_rerank_edges(img_bgr, edges_idx, nodes_dict, llm)

        return self._finalize(node_labels, node_confs, edges_idx, weights_list,
                              weight_confs, question, source="optionB")

    # ── Shared confidence + text builder ─────────────────────────────────────

    def _finalize(self, node_labels, node_confs, edges_idx, weights_list,
                  weight_confs, question, source):
        n_low = sum(1 for c in node_confs if c < 0.3)
        if node_confs and n_low / len(node_confs) >= 0.6:
            node_labels = [f"N{i}" for i in range(len(node_labels))]
            node_confs  = [1.0] * len(node_labels)
            label_type  = "structural_index"
        else:
            label_type = "image_text"

        n_unknown  = sum(1 for l in node_labels if not l or l in ("?", "NONE"))
        label_conf = float(np.mean(node_confs)) if node_confs else 0.0
        label_conf *= 1.0 - n_unknown / max(1, len(node_labels))

        if weights_list and any(w is not None for w in weights_list):
            weight_conf = float(np.mean(weight_confs)) if weight_confs else 1.0
            n_missing   = sum(1 for w in weights_list if w is None)
            weight_conf *= 1.0 - n_missing / max(1, len(weights_list))
        else:
            weight_conf = 1.0

        overall_conf = 0.6 * label_conf + 0.4 * weight_conf
        universal_text = self._build_text(node_labels, edges_idx, weights_list,
                                          overall_conf, label_type)
        aug_prompt = self._REASONER_PROMPT.format(
            universal_block=universal_text, question=question)

        cv_ok = bool(edges_idx) and overall_conf >= 0.20
        return {"cv_ok": cv_ok, "confidence": overall_conf,
                "prompt": aug_prompt if cv_ok else question, "source": source}

    @staticmethod
    def _build_text(node_labels, edges, edge_weights, confidence, label_type):
        import networkx as nx
        L = lambda i: (node_labels[i] if 0 <= i < len(node_labels)
                       and node_labels[i] not in ("", "?", "NONE") else f"N{i}")
        G = nx.Graph()
        for i in range(len(node_labels)):
            G.add_node(i)
        weighted = any(w is not None for w in edge_weights)
        for (u, v), w in zip(edges, edge_weights):
            G.add_edge(u, v, weight=(w if w is not None else 1.0))

        lines = ["GRAPH_PROPERTIES:",
                 f"  nodes: {G.number_of_nodes()}", f"  edges: {G.number_of_edges()}",
                 "  directed: false", f"  weighted: {str(weighted).lower()}",
                 f"LABEL_TYPE: {label_type}",
                 "NODES: [" + ", ".join(L(i) for i in G.nodes()) + "]",
                 "ADJACENCY_LIST:"]
        for n in G.nodes():
            nbrs = ([f"({L(m)}, {G[n][m]['weight']:g})" if weighted else L(m)]
                    for m in G.neighbors(n))
            lines.append(f"  {L(n)}: [" + ", ".join(x for sub in nbrs for x in sub) + "]")
        lines.append("EDGE_TABLE:")
        for u, v, d in G.edges(data=True):
            lines.append(f"  ({L(u)}, {L(v)}, {d['weight']:g})" if weighted
                         else f"  ({L(u)}, {L(v)})")
        lines.append("DEGREE: {" + ", ".join(f"{L(n)}:{G.degree(n)}" for n in G.nodes()) + "}")
        lines.append(f"EXTRACTION_CONFIDENCE: {confidence:.2f}")
        return "\n".join(lines)

    # ── Combined extraction ───────────────────────────────────────────────────

    def _extract(self, img_bgr, question, llm):
        cv_struct = self._run_cv_structure(img_bgr)
        if cv_struct is None:
            return {"cv_ok": False, "confidence": 0.0, "prompt": question, "source": "none"}

        raw_nodes, nodes_dict, _, edges_idx = cv_struct

        try:
            result_a = self._extract_optionA(img_bgr, raw_nodes, nodes_dict,
                                              edges_idx, question)
        except Exception as e:
            print(f"  [AkibMethod] Option A failed: {e}")
            result_a = {"cv_ok": False, "confidence": 0.0, "prompt": question, "source": "optionA"}

        try:
            result_b = self._extract_optionB(img_bgr, raw_nodes, nodes_dict,
                                              edges_idx, question, llm)
        except Exception as e:
            print(f"  [AkibMethod] Option B failed: {e}")
            result_b = {"cv_ok": False, "confidence": 0.0, "prompt": question, "source": "optionB"}

        above = [r for r in (result_a, result_b)
                 if r["cv_ok"] and r["confidence"] >= self.CONF_GATE]
        return (max(above, key=lambda r: r["confidence"]) if above
                else max((result_a, result_b), key=lambda r: r["confidence"]))

    # ── __call__ ─────────────────────────────────────────────────────────────

    def __call__(self, question_id, question, answer_type, subject, img, llm):
        import cv2

        inst = get_instruction_suffix(answer_type)

        img_rgb = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        try:
            result = self._extract(img_bgr, question, llm)
        except Exception as e:
            print(f"[AkibMethod] Extraction failed for ID {question_id}: {e}")
            result = {"cv_ok": False, "confidence": 0.0, "prompt": question}

        use_aug = result["cv_ok"] and result["confidence"] >= self.CONF_GATE
        final_question = result["prompt"] if use_aug else question

        return HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"## Question\n {final_question}\n" + GUIDE.format(INST=inst),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"},
                },
            ]
        )



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class VasudevMethod:
    """
    Orchestrates DSG, VISER, and SCAFFOLD methods.
    Mode toggled via environment variable 'VASUDEV_MODE' (options: 'dsg', 'viser', 'scaffold').
    """
    def __init__(self, sub_method=None):
        # Default to dsg for now
        self.sub_method = sub_method or os.getenv("VASUDEV_MODE", "dsg")

    def __call__(self, question_id, question, answer_type, subject, img, llm):
        # 1. Determine instruction suffix (standard DynaMath format)
        if answer_type == 'multiple choice':
            inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif answer_type == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

        processed_img = img.copy()
        processed_question = question

        # 2. Apply the selected logic
        if self.sub_method == "viser":
            processed_img = apply_viser_scaffolding(processed_img)
            processed_question = get_viser_prompt(question)
        
        elif self.sub_method == "scaffold":
            processed_img = apply_scaffold_coordinates(processed_img)
            processed_question = get_scaffold_prompt(question)
            
        elif self.sub_method == "dsg":
            # Multi-turn DSG grounding loop
            processed_question = run_dsg_loop(img, question, llm, encode_image)

        # 3. Build and return the multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": f"## Question\n {processed_question}\n" + GUIDE.format(INST=inst)
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(processed_img)}"}
                }
            ]
        )
        return message
