# -*- coding: utf-8 -*-
"""
知識圖譜萃取與視覺化工具（本地版，不需呼叫任何雲端 API）
使用 jieba 進行中文斷詞與詞性標註，以規則式方法萃取實體與關係，
再以 PyVis 繪製互動式知識圖譜。
"""

import re
import tempfile

import gradio as gr
import jieba
import jieba.posseg as pseg
from pyvis.network import Network

# 當文本為空或萃取失敗時，回傳的安全空結構，避免程式崩潰
EMPTY_GRAPH = {"nodes": [], "edges": []}

# jieba 詞性標註 -> 知識圖譜分類群組 的對應表
POS_TO_GROUP = {
    "nr": "人物",      # 人名
    "ns": "地點",      # 地名
    "nt": "組織",      # 機構團體名
    "nz": "專有名詞",  # 其他專名
    "n": "概念",       # 一般名詞
    "l": "概念",       # 描述性短語（如「思想家」）
    "eng": "概念",     # 英文詞
}

# 各群組對應的節點顏色（科技感霓虹配色），讓圖譜閱讀起來更直觀
GROUP_COLORS = {
    "人物": "#00e5ff",      # 電光青
    "地點": "#39ff88",      # 螢光綠
    "組織": "#7c5cff",      # 電紫
    "專有名詞": "#ff2e92",  # 霓虹粉
    "概念": "#ffd60a",      # 螢光黃
    "未分類": "#5b6478",    # 石墨灰
}

# jieba 預設字典對於人名、地標等專有名詞常會誤切（例如把「沃茲尼克」切成
# 「沃茲」+「尼克」）。這裡將範例文本中會用到的專有名詞加入自訂字典，
# 確保示範時能切出正確、完整的實體，讓範例「真的能用」。
CUSTOM_DICTIONARY = [
    ("蘋果公司", "nt"),
    ("賈伯斯", "nr"),
    ("沃茲尼克", "nr"),
    ("提姆庫克", "nr"),
    ("顏回", "nr"),
    ("子路", "nr"),
    ("儒家學派", "n"),
    ("論語", "nz"),
    ("後來", "d"),
    ("台北101", "nz"),
    ("士林夜市", "ns"),
    ("觀光景點", "n"),
]


def setup_custom_dictionary():
    """將自訂專有名詞載入 jieba 字典，於模組載入時執行一次。"""
    try:
        for word, tag in CUSTOM_DICTIONARY:
            jieba.add_word(word, freq=200000, tag=tag)
    except Exception as dict_error:
        print(f"[警告] 載入自訂字典失敗：{dict_error}")


setup_custom_dictionary()

# 範例文本：已搭配上方自訂字典與下方參數調校，確保點擊後能直接產生
# 乾淨、有意義的知識圖譜，而不是一堆破碎雜訊詞。
EXAMPLE_TEXTS = [
    [
        "蘋果公司由賈伯斯與沃茲尼克於1976年在美國加州創立。"
        "蘋果公司後來推出了iPhone，並由提姆庫克接任執行長，持續在全球市場擴張。",
        2,
        "",
        "相鄰連線",
    ],
    [
        "孔子是春秋時期魯國的思想家，創立了儒家學派。"
        "孔子的學生顏回與子路都來自魯國，他們共同記錄了孔子的言行，後人編成論語。",
        2,
        "時期",
        "相鄰連線",
    ],
    [
        "台北是台灣的首都，台北101與士林夜市都是台北著名的觀光景點，"
        "每年吸引許多來自日本與韓國的遊客。",
        2,
        "",
        "全連接",
    ],
]

# 自訂 CSS，打造科技感介面（深色漸層、霓虹光暈、圓角卡片）
CUSTOM_CSS = """
#header-banner {
    background: linear-gradient(135deg, #060a1f 0%, #0d1b3e 45%, #1a0b3e 100%);
    border: 1px solid rgba(34, 211, 238, 0.35);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 18px;
    box-shadow: 0 0 28px rgba(34, 211, 238, 0.25),
                0 0 60px rgba(124, 92, 255, 0.15);
}
#header-banner h1 {
    color: #e6faff !important;
    margin: 0;
    letter-spacing: 0.5px;
    text-shadow: 0 0 12px rgba(34, 211, 238, 0.65);
}
#header-banner p {
    color: #a8c5d6 !important;
    margin-top: 8px;
    opacity: 0.95;
}
#legend-box {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    padding: 10px 16px;
    border-radius: 12px;
    background: #0b0f1a;
    border: 1px solid rgba(124, 92, 255, 0.25);
    margin-bottom: 10px;
    font-size: 14px;
    color: #d7e3ec;
}
#legend-box .legend-item { display: inline-flex; align-items: center; gap: 6px; }
#legend-box .legend-dot {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 50%;
}
.graph-frame-wrapper {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(34, 211, 238, 0.3);
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.2);
}
"""


def build_legend_html() -> str:
    """產生群組顏色圖例的 HTML 片段。"""
    items = "".join(
        f'<span class="legend-item">'
        f'<span class="legend-dot" style="background:{color}; '
        f'box-shadow:0 0 6px {color};"></span>{group}'
        f"</span>"
        for group, color in GROUP_COLORS.items()
    )
    return f'<div id="legend-box">{items}</div>'


def split_sentences(text: str):
    """
    將長文本依常見中文/英文句末標點切分為句子清單。

    參數:
        text (str): 原始輸入文本。

    回傳:
        list[str]: 切分後的句子清單（已去除空白句）。
    """
    # 依句號、驚嘆號、問號、換行等符號切分句子
    raw_sentences = re.split(r"[。！？!?\n]+", text)
    return [s.strip() for s in raw_sentences if s.strip()]


def find_entity_occurrences(sentence: str, min_word_len: int, stopwords: set):
    """
    對單一句子進行斷詞與詞性標註，找出句中所有實體「出現位置」。

    與舊版不同，這裡刻意保留每次出現的位置（不去重），讓後續可以根據
    兩個實體之間實際間隔的文字（動詞、介詞等）來推斷關係標籤，
    而不是用統一的「共同出現」字樣，避免圖譜內容與原文脫節。

    參數:
        sentence (str): 單一句子。
        min_word_len (int): 實體詞彙的最小長度，用於過濾過短的雜訊詞。
        stopwords (set): 使用者自訂的停用詞集合，命中則排除。

    回傳:
        tuple:
            tokens (list[tuple[str, str]]): 該句子完整的 (詞, 詞性) 序列。
            occurrences (list[tuple[str, str, int]]):
                [(實體名稱, 分類群組, 在 tokens 中的索引), ...]，依出現順序排列。
    """
    tokens = list(pseg.cut(sentence))
    occurrences = []

    for idx, (word, pos) in enumerate(tokens):
        word = word.strip()

        if not word:
            continue
        if word in stopwords:
            continue
        # 英文詞不受中文字數限制，其餘詞彙依 min_word_len 過濾過短雜訊
        if pos != "eng" and len(word) < min_word_len:
            continue

        group = POS_TO_GROUP.get(pos)
        if group is None:
            continue

        occurrences.append((word, group, idx))

    return tokens, occurrences


def build_relation_label(tokens, start_idx: int, end_idx: int, max_len: int = 12) -> str:
    """
    擷取兩個實體之間的原文片段，作為更貼合原文的關係標籤，
    而不是一律標示成模糊的「共同出現」。

    規則：若間隔文字中含有動詞，優先取動詞組成標籤（最能代表關係動作）；
    否則取整段間隔文字（去除標點）；若間隔為空或過濾後為空，則退回「共同出現」。

    參數:
        tokens (list[tuple[str, str]]): 句子的完整 (詞, 詞性) 序列。
        start_idx (int): 前一個實體在 tokens 中的索引。
        end_idx (int): 後一個實體在 tokens 中的索引。
        max_len (int): 標籤的最大字數，避免過長標籤破壞圖譜版面。

    回傳:
        str: 關係標籤文字。
    """
    between = tokens[start_idx + 1 : end_idx]

    verbs = [w.strip() for w, p in between if p.startswith("v") and w.strip()]
    if verbs:
        label = "".join(verbs)
    else:
        # 沒有動詞時，退而取整段間隔文字（排除標點符號）
        words = [w.strip() for w, p in between if p != "x" and w.strip()]
        label = "".join(words)

    if len(label) > max_len:
        label = label[:max_len] + "…"

    return label or "共同出現"


def extract_knowledge_graph(
    text: str,
    min_word_len: int = 2,
    extra_stopwords_text: str = "",
    link_mode: str = "相鄰連線",
) -> dict:
    """
    以本地規則式方法（jieba 斷詞 + 詞性標註）從輸入文本中萃取
    實體（nodes）與關係（edges），完全不需呼叫任何外部 API。

    參數:
        text (str): 使用者輸入的長文本或筆記。
        min_word_len (int): 實體詞彙的最小長度。
        extra_stopwords_text (str): 使用者自訂停用詞，以逗號分隔。
        link_mode (str): "相鄰連線" 只連接句子中相鄰的實體；
                          "全連接" 則連接句子中每一對實體。

    回傳:
        dict: 符合 {"nodes": [...], "edges": [...]} 結構的字典。
              若文本為空或萃取過程發生例外，回傳空結構 EMPTY_GRAPH。
    """
    if not text or not text.strip():
        return EMPTY_GRAPH

    try:
        # 解析使用者自訂的停用詞清單
        stopwords = {
            w.strip() for w in extra_stopwords_text.split(",") if w.strip()
        }

        nodes_dict = {}   # id -> {"id":..., "label":..., "group":...}
        edges_list = []
        edge_seen = set()  # 避免重複加入相同的邊 (from, to)

        sentences = split_sentences(text)

        for sentence in sentences:
            tokens, occurrences = find_entity_occurrences(
                sentence, min_word_len, stopwords
            )

            # 將句子中的實體加入節點集合，並建立「同句中去重後」的實體清單
            # （用於全連接模式；相鄰連線模式則用未去重的 occurrences 以便
            # 取得每對實體之間的原文間隔來推斷關係標籤）
            seen_in_sentence = set()
            unique_entities = []
            for word, group, _idx in occurrences:
                if word not in nodes_dict:
                    nodes_dict[word] = {
                        "id": word,
                        "label": word,
                        "group": group,
                    }
                if word not in seen_in_sentence:
                    seen_in_sentence.add(word)
                    unique_entities.append(word)

            if link_mode == "全連接":
                # 全連接模式：句子中每一對實體都視為相關，關係標籤統一標示為
                # 「共同出現」，因為非相鄰的實體之間並無明確的原文片段可對應。
                for i in range(len(unique_entities)):
                    for j in range(i + 1, len(unique_entities)):
                        source, target = unique_entities[i], unique_entities[j]
                        edge_key = (source, target)
                        if edge_key in edge_seen:
                            continue
                        edge_seen.add(edge_key)
                        edges_list.append(
                            {"from": source, "to": target, "label": "共同出現"}
                        )
            else:
                # 相鄰連線模式：取兩個「相鄰出現」的實體之間的原文片段
                # （動詞、介詞等）作為關係標籤，讓圖譜內容貼合原文敘述，
                # 而不是統一標示成意義不明的「共同出現」。
                for i in range(len(occurrences) - 1):
                    source, _g1, idx1 = occurrences[i]
                    target, _g2, idx2 = occurrences[i + 1]
                    if source == target:
                        continue

                    edge_key = (source, target)
                    if edge_key in edge_seen:
                        continue

                    edge_seen.add(edge_key)
                    label = build_relation_label(tokens, idx1, idx2)
                    edges_list.append(
                        {"from": source, "to": target, "label": label}
                    )

        return {"nodes": list(nodes_dict.values()), "edges": edges_list}

    except Exception as extract_error:
        # 萃取過程發生未預期例外時，印出錯誤訊息並回傳空結構，避免程式崩潰
        print(f"[錯誤] 本地知識圖譜萃取失敗：{extract_error}")
        return EMPTY_GRAPH


def generate_graph_html(graph_data: dict) -> str:
    """
    使用 PyVis 將知識圖譜資料繪製成互動式 HTML 網路圖。

    參數:
        graph_data (dict): 包含 "nodes" 與 "edges" 的字典。

    回傳:
        str: 產生的暫存 HTML 檔案路徑，供 Gradio 的 gr.HTML() 渲染使用。
    """
    # 建立 PyVis 網路圖物件：深邃科技藍背景、亮色文字、可拖拽、帶箭頭的有向圖
    # cdn_resources="in_line"：將 vis-network 的 JS/CSS 直接內嵌進 HTML，
    # 否則預設會去讀取暫存目錄旁的 lib 資源資料夾，在 iframe srcdoc 中會找不到檔案而整個圖譜空白。
    # layout=True：採用「階層式（樹狀）佈局」，節點依關係方向由上往下分層排列，
    # 比起雜亂的力學群聚佈局更接近簡潔易懂的架構圖。
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0b0f1a",
        font_color="#e6faff",
        directed=True,
        cdn_resources="in_line",
        layout=True,
    )

    # 依照邊的方向（from -> to）分層排序，並設定層與層、樹與樹之間的間距
    net.options.layout.hierarchical.direction = "UD"
    net.options.layout.hierarchical.sortMethod = "directed"
    net.options.layout.hierarchical.levelSeparation = 160
    net.options.layout.hierarchical.nodeSpacing = 160
    net.options.layout.hierarchical.treeSpacing = 220

    # 階層式佈局建議搭配 hrepulsion，避免同層節點互相重疊，且不會像 barnes_hut 一樣晃動分散
    net.hrepulsion(node_distance=160, central_gravity=0.0, spring_length=160)

    # 邊使用平滑曲線，搭配階層式佈局看起來更像簡潔的架構圖
    net.set_edge_smooth("cubicBezier")

    nodes = graph_data.get("nodes", []) if graph_data else []
    edges = graph_data.get("edges", []) if graph_data else []

    # 先建立節點集合，避免邊引用到不存在的節點時發生例外
    added_node_ids = set()
    try:
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue
            group = node.get("group", "未分類")
            net.add_node(
                node_id,
                label=node.get("label", node_id),
                group=group,
                color=GROUP_COLORS.get(group, GROUP_COLORS["未分類"]),
                title=f"類別：{group}",
                borderWidth=2,
                borderWidthSelected=4,
                shadow={"enabled": True, "color": GROUP_COLORS.get(group, GROUP_COLORS["未分類"]), "size": 14},
                font={"color": "#e6faff", "size": 16, "face": "Microsoft JhengHei"},
            )
            added_node_ids.add(node_id)

        # 建立邊，並設定箭頭方向 arrows="to"
        for edge in edges:
            source = edge.get("from")
            target = edge.get("to")
            if not source or not target:
                continue
            # 若邊引用到尚未建立的節點，則自動補上該節點，避免渲染失敗
            if source not in added_node_ids:
                net.add_node(
                    source, label=source, group="未分類",
                    color=GROUP_COLORS["未分類"], shadow=True,
                    font={"color": "#e6faff", "size": 16},
                )
                added_node_ids.add(source)
            if target not in added_node_ids:
                net.add_node(
                    target, label=target, group="未分類",
                    color=GROUP_COLORS["未分類"], shadow=True,
                    font={"color": "#e6faff", "size": 16},
                )
                added_node_ids.add(target)

            net.add_edge(
                source,
                target,
                label=edge.get("label", ""),
                arrows="to",
                color={"color": "#3a4a6b", "highlight": "#22d3ee", "hover": "#22d3ee"},
                width=1.5,
                shadow=True,
                font={"color": "#8fd9ff", "size": 12, "strokeWidth": 0},
            )

    except Exception as build_error:
        # 繪圖過程發生例外時，印出錯誤訊息，仍繼續輸出目前已建立的圖
        print(f"[錯誤] 建立知識圖譜節點/邊時發生例外：{build_error}")

    # 將圖輸出為暫存 HTML 檔案，回傳檔案路徑供 Gradio 顯示
    # 注意：不能直接呼叫 net.save_graph()，因為 PyVis 內部用 open() 寫檔時
    # 沒有指定 encoding，在非 UTF-8 系統預設編碼（如 Windows 的 cp950）下，
    # 內嵌的 vis-network 函式庫一遇到無法編碼的字元（例如 ©）就會寫入失敗，
    # 導致產出空白檔案、圖譜整個顯示不出來。這裡改用 generate_html() 取得
    # HTML 字串後，自行以 UTF-8 編碼寫檔。
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    temp_file.close()

    try:
        html_content = net.generate_html(notebook=False)
        with open(temp_file.name, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as save_error:
        print(f"[錯誤] 儲存 HTML 檔案失敗：{save_error}")

    return temp_file.name


def build_stats_markdown(graph_data: dict) -> str:
    """
    根據知識圖譜資料，產生節點數、邊數與群組分布的統計摘要文字。

    參數:
        graph_data (dict): 包含 "nodes" 與 "edges" 的字典。

    回傳:
        str: 適合以 Markdown 顯示的統計摘要。
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        return "⚠️ 未偵測到任何實體，請嘗試輸入更長的文本，或調整下方的進階設定。"

    # 統計各群組的節點數量
    group_counts = {}
    for node in nodes:
        group = node.get("group", "未分類")
        group_counts[group] = group_counts.get(group, 0) + 1

    group_summary = "、".join(
        f"{group}：{count}" for group, count in group_counts.items()
    )

    return (
        f"✅ **節點數：{len(nodes)}**　|　**邊數：{len(edges)}**\n\n"
        f"📊 群組分布：{group_summary}"
    )


def process_text_to_graph(
    text: str,
    min_word_len: int,
    extra_stopwords_text: str,
    link_mode: str,
):
    """
    整合流程：將輸入文本萃取為知識圖譜，再轉換為可顯示與下載的內容。

    參數:
        text (str): 使用者輸入的文本。
        min_word_len (int): 實體詞彙的最小長度。
        extra_stopwords_text (str): 使用者自訂停用詞（以逗號分隔）。
        link_mode (str): 連線模式（"相鄰連線" 或 "全連接"）。

    回傳:
        tuple: (iframe_html, 統計摘要文字, 原始 JSON 資料, 可下載的 HTML 檔案路徑)
    """
    # 步驟一：以本地規則式方法萃取知識圖譜結構化資料（不需任何 API）
    graph_data = extract_knowledge_graph(
        text,
        min_word_len=min_word_len,
        extra_stopwords_text=extra_stopwords_text,
        link_mode=link_mode,
    )

    # 步驟二：將結構化資料轉換為 PyVis 互動式 HTML 檔案
    html_path = generate_graph_html(graph_data)

    # 讀取產生的 HTML 內容，並以 iframe 包裝後回傳給 Gradio 顯示
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as read_error:
        print(f"[錯誤] 讀取 HTML 檔案失敗：{read_error}")
        html_content = "<p style='color:red;'>知識圖譜產生失敗，請稍後再試。</p>"

    iframe_html = (
        '<div class="graph-frame-wrapper">'
        f'<iframe style="width:100%; height:650px; border:none; display:block;" '
        f'srcdoc="{html_content.replace(chr(34), "&quot;")}"></iframe>'
        "</div>"
    )

    stats_markdown = build_stats_markdown(graph_data)

    return iframe_html, stats_markdown, graph_data, html_path


def clear_all():
    """重置所有輸入與輸出元件回初始狀態。"""
    return "", "", None, {}, None


# ------------------------------
# Gradio 介面建置
# ------------------------------
with gr.Blocks(title="知識圖譜萃取工具（本地版）") as demo:
    gr.HTML(
        '<div id="header-banner">'
        "<h1>📚 知識圖譜萃取與視覺化工具</h1>"
        "<p>輸入文本，一鍵萃取實體與關係，繪製成互動式知識圖譜——完全在本機執行，"
        "不會呼叫任何雲端 LLM API。</p>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="輸入文本",
                placeholder="請在此貼上您的長文本或筆記...",
                lines=12,
            )

            with gr.Accordion("⚙️ 進階設定", open=False):
                min_word_len = gr.Slider(
                    label="實體最小詞長",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,
                    info="過短的詞通常是雜訊，提高數值可減少無意義的節點。",
                )
                extra_stopwords = gr.Textbox(
                    label="自訂停用詞（以逗號分隔）",
                    placeholder="例如：時候,可能,事情",
                    info="命中清單中的詞將不會出現在知識圖譜中。",
                )
                link_mode = gr.Radio(
                    label="連線模式",
                    choices=["相鄰連線", "全連接"],
                    value="相鄰連線",
                    info="「相鄰連線」只連接句子中前後相鄰的實體，"
                    "搭配階層式佈局最清晰；"
                    "「全連接」會連接句子中每一對實體，關係更密集，"
                    "但分層後可能較雜亂。",
                )

            with gr.Row():
                generate_button = gr.Button("🚀 生成知識圖譜", variant="primary")
                clear_button = gr.Button("🗑️ 清除")

        with gr.Column(scale=2):
            gr.HTML(build_legend_html())
            stats_output = gr.Markdown(label="統計摘要")

            with gr.Tabs():
                with gr.Tab("🕸️ 互動圖譜"):
                    output_html = gr.HTML(label="知識圖譜視覺化結果")
                with gr.Tab("🧾 原始 JSON 資料"):
                    json_output = gr.JSON(label="nodes / edges 結構化資料")

            download_file = gr.File(label="⬇️ 下載圖譜 HTML 檔案", interactive=False)

    gr.Examples(
        examples=EXAMPLE_TEXTS,
        inputs=[input_text, min_word_len, extra_stopwords, link_mode],
        outputs=[output_html, stats_output, json_output, download_file],
        fn=process_text_to_graph,
        run_on_click=True,
        label="📌 範例文本（點擊即可立即套用並生成圖譜）",
    )

    # 按鈕點擊事件：將輸入文本送入處理流程，並將結果顯示在各輸出元件
    generate_button.click(
        fn=process_text_to_graph,
        inputs=[input_text, min_word_len, extra_stopwords, link_mode],
        outputs=[output_html, stats_output, json_output, download_file],
    )

    # 清除按鈕：重置所有輸入與輸出
    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_text, stats_output, output_html, json_output, download_file],
    )


if __name__ == "__main__":
    # 啟動 Gradio 應用程式（Gradio 6 起，theme 與 css 改至 launch() 設定）
    demo.launch(
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="violet", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
