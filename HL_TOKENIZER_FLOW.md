# HLTokenizer Flowchart v3.2 (Preprocess Tag → Batch Trans → Finalize)

```mermaid
graph TD
    A[Input Text / 輸入文本] --> B[Lang Segment / 語言區塊分割<br/>regex: CH | Latin | JA | other]
    B --> C{ZH block? / 中文塊?}
    C -->|Yes| D[jieba.lcut / 中文分詞]
    C -->|No| E{Ja? / 日文?}
    E -->|Yes| F[Janome tokenize / 日文詞性濾]
    E -->|No| G[Regex words, lower EN-like / 詞邊界，低寫英文]
    D --> H[Words list / 詞列表]
    F --> H
    G --> H
    H --> I{In hl_vocab? / 詞彙中?}
    I -->|Yes| J[[HL你好] or [HL你好:en]]
    I -->|No| K{hash fallback?}
    K -->|Yes| L[[Hxxxx]]
    K -->|No| M[orig word / 原詞]
    J --> N[Output tokens / 輸出 token]
    L --> N
    M --> N

    subgraph \"Vocab Build / 詞彙建構\"
        O[Manual Concepts / 手動概念<br/>'apple':['苹果','apple']] --> P[zh_base='苹果'<br/>[HL苹果], variants [HL苹果:en]]
        Texts --> Q[Tokenize cross-lang / 跨語言分詞]
        Q --> R[Unique new words / 新獨特詞]
        R --> S{non-zh? / 非中文?}
        S -->|Yes| T[trans/cache → zh_pivot / 翻譯樞紐]
        S -->|No| U[zh as-is / 中文原樣]
        T --> V[zh_counts freq / 中樞紐頻次]
        U --> V
        V --> W[Top zh → [HLzh] + map variants / 高頻中樞紐 ID]
    end

    subgraph \"Decode / 解碼\"
        X[Tokens split / 分詞] --> Y{[HLbase:lang]?}
        Y -->|Yes| Z[back-trans / variant_rev<br/>[HL你好:en] → hello]
        Y -->|No| AA[zh base / 中文基底]
        X -->|Hash/orig| BB[as-is / 原樣]
        Z --> CC[Output mix / 混合輸出]
        AA --> CC
        BB --> CC
    end

    style P fill:#bbf
    style W fill:#bbf
    style Z fill:#f9f
```

## Key Updates v2.4 / 關鍵更新：
- **Per-segment tokenize**: No mixed-lang glue (pangram split ✓).
- **Encode**: Word → shared [HLID].
- **Vocab**: Freq + manual → zh pivot.
- **Decode**: Dynamic back-trans.

**Render in Markdown viewer! / 複製至 Markdown 檢視器渲染！**