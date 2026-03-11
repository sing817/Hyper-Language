# HLTokenizer 流程圖 (Mermaid)

```mermaid
graph TD
    A[輸入 Text] --> B{中文?}
    B -->|是| C[jieba cut]
    B -->|否| D[regex \\b\\w+\\b]
    C --> E[words lower? no for zh]
    D --> E
    E --> F{word in hl_vocab?}
    F -->|是| G[[HLxxxx]]
    F -->|否| H{hash fallback?}
    H -->|是| I[[Hxxxx]]
    H -->|否| J[orig word]
    G --> K[Output tokens]
    I --> K
    J --> K

    subgraph "Vocab Build (build_vocab)"
        L[Texts] --> M[Tokenize all]
        M --> N[Unique words]
        N --> O{non-zh?}
        O -->|是| P[Qwen translate to zh single word<br/>batch chat prompt]
        O -->|否| Q[zh as is]
        P --> R[zh_rep map]
        Q --> R
        R --> S[Counter zh_rep freq]
        S --> T[Top freq → [HLID]<br/>rep = zh_rep]
    end

    U[Tokens] --> V{decode?}
    V -->| [HL/H ]| W[reverse_hl rep zh word]
    V -->|orig| X[orig]
    W --> Y[Output zh reps]
    X --> Y

    style P fill:#f9f,stroke:#333
    style T fill:#bbf,stroke:#333
```

## 關鍵：
- **Encode**：word → [HLID] (shared via zh pivot) or hash。
- **Vocab**：non-zh → zh trans (Qwen)，zh native → freq zh → ID。
- **Decode**：ID → zh rep (abstract Chinese)。
- **Compression**：multi word → 1 [HLID]。

Copy to Markdown viewer render！