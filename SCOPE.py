import torch
from torch import nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@MODELS.register_module(name="SCOPESegmentation")
class SCOPESegmentation(BaseSegmentor):
    """SCOPE for training-free open-vocabulary segmentation.

    This implementation follows the paper's three inference-time modules:
      1. Semantic-Adaptive Branch (SAB)
      2. Collaborative Dual-Head Decoding (CDHD)
      3. Multi-View Consensus (MVC)

    The class is registered under both ``SCOPESegmentation`` and the legacy
    name ``SegEarthOV3Segmentation`` for configuration compatibility.
    """

    def __init__(
        self,
        classname_path,
        device=torch.device("cuda"),
        prob_thd=0.0,
        bg_idx=0,
        slide_stride=0,
        slide_crop=0,
        confidence_threshold=0.5,
        use_semantic_head=True,
        use_instance_head=True,
        apply_presence_scaling=True,
        use_sab=True,
        sab_beta=0.15,
        sab_presence_tau=0.35,
        sab_uncertainty_tau=0.55,
        sab_fg_tau=0.60,
        sab_topk_frac=0.01,
        sab_topk_min=512,
        use_cdhd=True,
        cdhd_beta=0.35,
        cdhd_ref_tau=0.25,
        cdhd_gamma=2.0,
        cdhd_support_tau=0.10,
        cdhd_topk_instances=15,
        cdhd_min_area=64,
        cdhd_bin_tau=0.0,
        cdhd_only_improve=True,
        use_mvc=True,
        data_preprocessor=None,
        init_cfg=None,
        model_type=None,
        **legacy_kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self._apply_legacy_kwargs(legacy_kwargs)

        use_semantic_head = legacy_kwargs.pop("use_semantic_head", use_semantic_head)
        use_instance_head = legacy_kwargs.pop("use_instance_head", use_instance_head)
        apply_presence_scaling = legacy_kwargs.pop(
            "apply_presence_scaling", apply_presence_scaling
        )
        use_cdhd = legacy_kwargs.pop("use_cdhd", use_cdhd)
        cdhd_beta = legacy_kwargs.pop("cdhd_beta", cdhd_beta)
        cdhd_ref_tau = legacy_kwargs.pop("cdhd_ref_tau", cdhd_ref_tau)
        cdhd_gamma = legacy_kwargs.pop("cdhd_gamma", cdhd_gamma)
        cdhd_support_tau = legacy_kwargs.pop("cdhd_support_tau", cdhd_support_tau)
        cdhd_topk_instances = legacy_kwargs.pop(
            "cdhd_topk_instances", cdhd_topk_instances
        )
        cdhd_min_area = legacy_kwargs.pop("cdhd_min_area", cdhd_min_area)
        cdhd_bin_tau = legacy_kwargs.pop("cdhd_bin_tau", cdhd_bin_tau)
        cdhd_only_improve = legacy_kwargs.pop(
            "cdhd_only_improve", cdhd_only_improve
        )
        use_mvc = legacy_kwargs.pop("use_mvc", use_mvc)
        use_sab = legacy_kwargs.pop("use_sab", use_sab)
        sab_beta = legacy_kwargs.pop("sab_beta", sab_beta)
        sab_presence_tau = legacy_kwargs.pop("sab_presence_tau", sab_presence_tau)
        sab_uncertainty_tau = legacy_kwargs.pop(
            "sab_uncertainty_tau", sab_uncertainty_tau
        )
        sab_fg_tau = legacy_kwargs.pop("sab_fg_tau", sab_fg_tau)
        sab_topk_frac = legacy_kwargs.pop("sab_topk_frac", sab_topk_frac)
        sab_topk_min = legacy_kwargs.pop("sab_topk_min", sab_topk_min)

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self.device = device
        self.processor = Sam3Processor(
            build_sam3_image_model(
                bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                checkpoint_path="weights/sam3/sam3.pt",
                device="cuda",
            ),
            confidence_threshold=confidence_threshold,
            device=device,
        )

        self.query_words, query_idx = get_cls_idx(classname_path)
        self.num_cls = max(query_idx) + 1
        self.num_queries = len(query_idx)
        self.query_idx = torch.tensor(query_idx, dtype=torch.int64, device=device)
        self._ddp_dummy = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=True
        )

        self.prob_thd = float(prob_thd)
        self.bg_idx = int(bg_idx)
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        self.use_semantic_head = bool(use_semantic_head)
        self.use_instance_head = bool(use_instance_head)
        self.apply_presence_scaling = bool(apply_presence_scaling)

        self.use_sab = bool(use_sab)
        self.sab_beta = float(sab_beta)
        self.sab_presence_tau = float(sab_presence_tau)
        self.sab_uncertainty_tau = float(sab_uncertainty_tau)
        self.sab_fg_tau = float(sab_fg_tau)
        self.sab_topk_frac = float(sab_topk_frac)
        self.sab_topk_min = int(sab_topk_min)

        self.use_cdhd = bool(use_cdhd)
        self.cdhd_beta = float(cdhd_beta)
        self.cdhd_ref_tau = float(cdhd_ref_tau)
        self.cdhd_gamma = float(cdhd_gamma)
        self.cdhd_support_tau = float(cdhd_support_tau)
        self.cdhd_topk_instances = int(cdhd_topk_instances)
        self.cdhd_min_area = int(cdhd_min_area)
        self.cdhd_bin_tau = float(cdhd_bin_tau)
        self.cdhd_only_improve = bool(cdhd_only_improve)

        self.use_mvc = bool(use_mvc)

    @staticmethod
    def _apply_legacy_kwargs(legacy_kwargs):
        """Map legacy experiment-time argument names to SCOPE names."""
        rename_map = {
            "use_sem_seg": "use_semantic_head",
            "use_transformer_decoder": "use_instance_head",
            "use_presence_score": "apply_presence_scaling",
            "use_tta": "use_mvc",
            "use_sfr": "use_cdhd",
            "sfr_beta": "cdhd_beta",
            "sfr_tau": "cdhd_ref_tau",
            "sfr_gamma": "cdhd_gamma",
            "sfr_support_thd": "cdhd_support_tau",
            "sfr_topk_instances": "cdhd_topk_instances",
            "sfr_min_area": "cdhd_min_area",
            "sfr_bin_thd": "cdhd_bin_tau",
            "sfr_only_improve": "cdhd_only_improve",
        }

        for old_key, new_key in rename_map.items():
            if old_key in legacy_kwargs and new_key not in legacy_kwargs:
                legacy_kwargs[new_key] = legacy_kwargs.pop(old_key)

    def _to_probability(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return tensor
        if not tensor.dtype.is_floating_point:
            return tensor.float()
        tensor_min = float(tensor.min().detach().cpu())
        tensor_max = float(tensor.max().detach().cpu())
        if tensor_min < 0.0 or tensor_max > 1.0:
            return tensor.sigmoid()
        return tensor

    def _resize_map(self, tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if tensor.shape == (h, w):
            return tensor
        return F.interpolate(
            tensor.view(1, 1, *tensor.shape),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    def _presence_scalar(self, presence_score) -> torch.Tensor:
        if presence_score is None:
            return torch.tensor(0.0, device=self.device)
        if torch.is_tensor(presence_score):
            return presence_score.float().mean()
        return torch.tensor(float(presence_score), device=self.device)

    def _compute_sab_uncertainty(self, semantic_logits, presence_score) -> torch.Tensor:
        eps = 1e-6
        ln2 = 0.693147

        if self.use_semantic_head and semantic_logits is not None:
            semantic_prob = semantic_logits
            if semantic_prob.dim() == 4:
                semantic_prob = semantic_prob.squeeze(0).squeeze(0).float()
            else:
                semantic_prob = semantic_prob.float()

            semantic_prob = semantic_prob.clamp(eps, 1.0 - eps)
            roi = semantic_prob > self.sab_fg_tau
            entropy = -(
                semantic_prob * torch.log(semantic_prob)
                + (1.0 - semantic_prob) * torch.log(1.0 - semantic_prob)
            )
            values = entropy[roi] if roi.any().item() else entropy.reshape(-1)

            num_values = values.numel()
            topk = int(num_values * self.sab_topk_frac)
            topk = max(self.sab_topk_min, topk)
            topk = min(topk, num_values)
            topk_entropy = torch.topk(values, topk, largest=True).values.mean()
            return (topk_entropy / ln2).clamp(0.0, 1.0)

        presence_prob = presence_score.clamp(0.0, 1.0)
        return (1.0 - (presence_prob - 0.5).abs() * 2.0).clamp(0.0, 1.0)

    def _run_sab(self, inference_state, query_word):
        """Semantic-Adaptive Branch with conditional re-inference."""
        inference_state = self.processor.set_text_prompt(
            state=inference_state,
            prompt=query_word,
        )

        if not self.use_sab:
            return inference_state

        pass1_semantic = inference_state.get("semantic_mask_logits", None)
        pass1_masks = inference_state.get("masks_logits", None)
        pass1_scores = inference_state.get("object_score", None)
        pass1_presence = inference_state.get("presence_score", None)

        pass1_presence_t = self._presence_scalar(pass1_presence)
        uncertainty = self._compute_sab_uncertainty(pass1_semantic, pass1_presence_t)

        do_adapt = (
            pass1_presence_t.item() > self.sab_presence_tau
            and uncertainty.item() > self.sab_uncertainty_tau
        )
        if not do_adapt:
            return inference_state

        backbone_out = inference_state.get("backbone_out", None)
        if backbone_out is None:
            return inference_state

        backbone_fpn = backbone_out.get("backbone_fpn", None)
        if not isinstance(backbone_fpn, (list, tuple)) or len(backbone_fpn) < 3:
            return inference_state

        eps = 1e-6
        raw_weights = torch.stack(
            [
                0.5 + uncertainty,
                torch.tensor(0.5, device=self.device),
                0.5 + (1.0 - uncertainty),
            ]
        )
        weights = (raw_weights / raw_weights.sum()).clamp(eps, 1.0)
        scales = (1.0 + self.sab_beta * (3.0 * weights - 1.0)).clamp(
            1.0 - self.sab_beta,
            1.0 + 2.0 * self.sab_beta,
        )

        original_fpn = backbone_fpn
        try:
            reweighted_fpn = []
            for level in range(3):
                scale = scales[level].to(
                    dtype=original_fpn[level].dtype,
                    device=original_fpn[level].device,
                )
                reweighted_fpn.append(original_fpn[level] * scale)
            backbone_out["backbone_fpn"] = reweighted_fpn
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt=query_word,
            )
        finally:
            backbone_out["backbone_fpn"] = original_fpn

        if self.use_semantic_head and pass1_semantic is not None:
            pass2_semantic = inference_state.get("semantic_mask_logits", None)
            if pass2_semantic is not None:
                inference_state["semantic_mask_logits"] = torch.max(
                    pass1_semantic,
                    pass2_semantic,
                )

        pass2_masks = inference_state.get("masks_logits", None)
        pass2_scores = inference_state.get("object_score", None)
        if (
            pass1_masks is not None
            and pass2_masks is not None
            and pass1_masks.numel() > 0
            and pass2_masks.numel() > 0
        ):
            inference_state["masks_logits"] = torch.cat([pass1_masks, pass2_masks], dim=0)
            if (
                pass1_scores is not None
                and pass2_scores is not None
                and pass1_scores.numel() > 0
                and pass2_scores.numel() > 0
            ):
                inference_state["object_score"] = torch.cat(
                    [pass1_scores, pass2_scores], dim=0
                )
        elif pass1_masks is not None and pass2_masks is not None and pass2_masks.numel() == 0:
            inference_state["masks_logits"] = pass1_masks
            if pass1_scores is not None:
                inference_state["object_score"] = pass1_scores

        pass2_presence = inference_state.get("presence_score", None)
        if pass2_presence is not None:
            pass2_presence_t = self._presence_scalar(pass2_presence)
            inference_state["presence_score"] = torch.max(pass1_presence_t, pass2_presence_t)

        return inference_state

    def _build_instance_response(self, inference_state, h: int, w: int) -> torch.Tensor:
        instance_response = torch.zeros((h, w), device=self.device)
        masks_logits = inference_state.get("masks_logits", None)
        object_score = inference_state.get("object_score", None)

        if not self.use_instance_head or masks_logits is None or object_score is None:
            return instance_response
        if masks_logits.shape[0] == 0:
            return instance_response

        for instance_id in range(int(masks_logits.shape[0])):
            instance_logits = masks_logits[instance_id].squeeze()
            instance_logits = self._resize_map(instance_logits, h, w)
            instance_response = torch.maximum(
                instance_response,
                instance_logits * object_score[instance_id],
            )

        return instance_response

    def _build_semantic_response(self, inference_state, h: int, w: int):
        semantic_logits = inference_state.get("semantic_mask_logits", None)
        if not self.use_semantic_head or semantic_logits is None:
            return None
        if semantic_logits.shape != (h, w):
            semantic_logits = F.interpolate(
                semantic_logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        return semantic_logits

    def _run_cdhd(
        self,
        qprob: torch.Tensor,
        semantic_prob: torch.Tensor,
        masks_logits: torch.Tensor,
        object_score: torch.Tensor,
        h: int,
        w: int,
        presence_score=None,
    ) -> torch.Tensor:
        """Collaborative Dual-Head Decoding with local asymmetric refinement."""
        if not self.use_cdhd or masks_logits is None or object_score is None:
            return qprob
        if masks_logits.numel() == 0:
            return qprob

        feedback_map = semantic_prob if semantic_prob is not None else qprob
        feedback_map = self._to_probability(feedback_map)

        scores = object_score.view(-1)
        num_instances = int(masks_logits.shape[0])
        topk = min(num_instances, max(1, self.cdhd_topk_instances))
        if topk < num_instances:
            top_indices = torch.topk(scores, k=topk, largest=True).indices
        else:
            top_indices = torch.arange(num_instances, device=scores.device)

        prior = torch.zeros_like(qprob)
        apply_mask = torch.zeros_like(qprob, dtype=torch.bool)
        eps = 1e-6

        for idx in top_indices.tolist():
            mask_logits = masks_logits[idx].squeeze()
            mask_logits = self._resize_map(mask_logits, h, w)
            mask_prob = self._to_probability(mask_logits)

            binary_mask = mask_prob > self.cdhd_bin_tau
            area = int(binary_mask.sum().detach().cpu())
            if area < self.cdhd_min_area:
                continue

            support = feedback_map[binary_mask].mean()
            if float(support.detach().cpu()) < self.cdhd_support_tau:
                continue

            scale = ((support + eps) / (self.cdhd_ref_tau + eps)).pow(self.cdhd_gamma)
            scale = scale.clamp(0.25, 2.5)
            adjusted_score = (scores[idx] * scale).clamp(0.0, 1.0)
            prior = torch.maximum(prior, mask_prob * adjusted_score)
            apply_mask |= binary_mask

        if not bool(apply_mask.any().detach().cpu()):
            return qprob

        beta = float(self.cdhd_beta)
        if presence_score is not None:
            try:
                beta *= float(self._presence_scalar(presence_score).detach().cpu().item())
            except Exception:
                pass
        beta = float(max(0.0, min(1.0, beta)))

        refined = qprob + beta * (prior - qprob)
        if self.cdhd_only_improve:
            refined = torch.maximum(qprob, refined)
        return torch.where(apply_mask, refined, qprob)

    def _inference_single_view(self, image):
        """Run full SCOPE inference for a single view."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

            for query_id, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self._run_sab(inference_state, query_word)

                instance_response = self._build_instance_response(inference_state, h, w)
                semantic_response = self._build_semantic_response(inference_state, h, w)

                query_response = instance_response
                if semantic_response is not None:
                    query_response = torch.maximum(query_response, semantic_response)

                query_response = self._run_cdhd(
                    qprob=query_response,
                    semantic_prob=semantic_response,
                    masks_logits=inference_state.get("masks_logits", None),
                    object_score=inference_state.get("object_score", None),
                    h=h,
                    w=w,
                    presence_score=inference_state.get("presence_score", None),
                )

                if self.apply_presence_scaling:
                    query_response = (
                        query_response * inference_state["presence_score"]
                    )

                seg_logits[query_id] = query_response

        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Run sliding-window inference on a PIL image."""
        w_img, h_img = image.size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_logits = self._inference_single_view(crop_img)
                preds[:, y1:y2, x1:x2] += crop_logits
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Sparse sliding-window coverage detected."
        return preds / count_mat

    def _predict_view(self, image, ori_shape):
        if self.slide_crop > 0 and (
            self.slide_crop < image.size[0] or self.slide_crop < image.size[1]
        ):
            view_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
        else:
            view_logits = self._inference_single_view(image)

        if view_logits.shape[-2:] != ori_shape:
            view_logits = F.interpolate(
                view_logits.unsqueeze(0),
                size=ori_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return view_logits

    def _aggregate_synonyms(self, seg_logits: torch.Tensor) -> torch.Tensor:
        if self.num_cls == self.num_queries:
            return seg_logits
        seg_logits = seg_logits.unsqueeze(0)
        cls_index = nn.functional.one_hot(self.query_idx)
        cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
        return (seg_logits * cls_index).max(1)[0]

    def predict(self, inputs, data_samples):
        batch_img_metas = [data_sample.metainfo for data_sample in data_samples]

        for i, meta in enumerate(batch_img_metas):
            image_path = meta.get("img_path")
            ori_image = Image.open(image_path).convert("RGB")
            ori_shape = meta["ori_shape"]

            original_logits = self._predict_view(ori_image, ori_shape)
            if self.use_mvc:
                flipped_image = ori_image.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_logits = self._predict_view(flipped_image, ori_shape)
                flipped_logits = torch.flip(flipped_logits, dims=[2])
                seg_logits = torch.stack([original_logits, flipped_logits]).mean(dim=0)
            else:
                seg_logits = original_logits

            seg_logits = self._aggregate_synonyms(seg_logits)
            seg_pred = torch.argmax(seg_logits, dim=0)
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(data=seg_logits),
                    "pred_sem_seg": PixelData(data=seg_pred.unsqueeze(0)),
                }
            )

        return data_samples

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError("SCOPE implements inference-only segmentation.")

    def inference(self, img, batch_img_metas):
        raise NotImplementedError("Use predict() for inference.")

    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError("Use predict() for inference.")

    def extract_feat(self, inputs):
        raise NotImplementedError("SCOPE does not expose a training feature path.")

    def loss(self, inputs, data_samples):
        raise NotImplementedError("SCOPE is defined for training-free inference only.")


def get_cls_idx(path):
    with open(path, "r") as f:
        name_sets = f.readlines()

    class_names = []
    class_indices = []
    for class_id, name_set in enumerate(name_sets):
        names = [name.strip() for name in name_set.split(",")]
        class_names.extend(name.replace("\n", "") for name in names)
        class_indices.extend([class_id] * len(names))

    return class_names, class_indices
