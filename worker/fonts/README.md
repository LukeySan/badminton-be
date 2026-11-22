# Caption fonts

Bundled so exported reels look the same on every worker (Mac, CUDA PC, Modal),
whatever fonts the machine has. Used by `worker/caption_image.py`.

| File | Font | Source | License |
|---|---|---|---|
| `Manrope-ExtraBold.ttf` | Manrope ExtraBold (800), the app's font and weight for captions | googlefonts/manrope `fonts/ttf/manrope-extrabold.ttf` at commit 6f81ebe (the commit Google Fonts ships) | SIL OFL 1.1, `Manrope-OFL.txt` |
| `NotoSansThai-ExtraBold.ttf` | Noto Sans Thai ExtraBold (800): fallback for characters Manrope doesn't have (Thai) | notofonts/thai release NotoSansThai-v2.002, `NotoSansThai/googlefonts/ttf/` | SIL OFL 1.1, `NotoSansThai-OFL.txt` |

Unmodified. The OFL allows bundling them with software; the license files must stay
next to the fonts.
