# 🖼️ CLIP-Based Image Captioning & Text-to-Image Retrieval

This project implements an image captioning system trained on the COCO dataset using a **frozen CLIP image encoder** and a **Transformer-based decoder**. It also supports **text-to-image retrieval** using the dual-encoder nature of CLIP.

---

## 🚀 Features

- ✅ Frozen CLIP image encoder (ViT-B/32, LAION2B pretrained)
- 🧠 Transformer decoder trained to generate captions
- 🔍 Text-to-image retrieval using CLIP text & image encoders
- 🧩 COCO 2014 compatible (`train2014`, `val2014`)