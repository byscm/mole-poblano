# Models, licences and credits

This project **does not redistribute any model weights**. Each user downloads them from the official source and accepts their licence. The table summarises a review made in November 2026; licences change, so check the links yourself. This is not legal advice.

✅ commercial use allowed · ⚠️ allowed with conditions or unconfirmed · ⛔ non-commercial

| Model | Used for | Licence | Commercial | Link |
|---|---|---|---|---|
| SAM 3.1 (`sam3.1_multiplex_fp16`) | masks of floor, walls and objects | SAM License (Meta) | ⚠️ no military, weapons, surveillance or ITAR use | [GitHub](https://github.com/facebookresearch/sam3) |
| MoGe-2 ViT-L normal | depth, camera height | MIT | ✅ | [GitHub](https://github.com/microsoft/MoGe) |
| Qwen3-VL-8B | naming objects, questions about the photo | Apache-2.0 | ✅ | [HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| TRELLIS.2-4B | generating 3D models | MIT | ⚠️ depends on DINOv3 | [GitHub](https://github.com/microsoft/TRELLIS.2) |
| DINOv3 ViT-L (+ NAF) | TRELLIS.2 encoder, gallery matching | DINOv3 License (Meta) | ⚠️ show "Built with DINOv3"; same use restrictions as SAM; NAF weights licence unconfirmed | [Licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license) |
| Qwen-Image 2.1 | optional "clean view" and textures (`clean_view.py`, `texture_gen.py`) | Qwen Research License | ⛔ research only | [HF](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| Orient Anything V2 | which way each object faces (`orientation.py`), `vggt_align.py` | CC-BY-4.0, but derived from VGGT-1B (CC-BY-NC-4.0) | ⛔ treat as non-commercial | [GitHub](https://github.com/SpatialVision/Orient-Anything-V2) |
| SpatialLM 1.1 | optional second opinion on the room layout | CC-BY-NC-4.0 | ⛔ | [HF](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B) |
| LingBot-Map | optional video / multi-photo alignment | code Apache-2.0; weights licence not stated explicitly | ⚠️ | [GitHub](https://github.com/Robbyant/lingbot-map) |
| MapAnything | optional multi-photo alignment | Apache-2.0 variant (check the checkpoint you download) | ⚠️ | [GitHub](https://github.com/facebookresearch/map-anything) |
| Hi3D | optional mesh refinement (`refine_hi3d.py`) | not reviewed | ⚠️ check before use | — |

### Commercial use without the ⛔ models

- **Orientation:** the pipeline can fall back to the geometric yaw or to the VLM (`facing.py`). `eval_facing.py` measures the difference.
- **Clean view:** skip it. `generate_models.py` already generates from the SAM crop.
- **SpatialLM:** it is optional. `room_layout.py` (SAM + MoGe) already gives the walls, floor and ceiling.

## Software

- **ComfyUI** — GPL-3.0. The pipeline talks to it over its local HTTP API.
- **Blender** — GPL. The scripts that import `bpy` are GPL, and so is this repository.
- **three.js** r185 (MIT), bundled in `viewer/lib/`.
- numpy, Pillow, scipy (BSD / MIT-CMU).

## Credits

Built with DINOv3. Thanks to the authors of SAM, MoGe, Qwen, TRELLIS, Orient Anything, SpatialLM, LingBot-Map, MapAnything, Hi3D, ComfyUI, Blender and three.js.
