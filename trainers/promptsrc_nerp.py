import os
import os.path as osp
import gzip
import torch
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from dassl.engine import TRAINER_REGISTRY

from clip import clip
from collections import Counter
import json
import copy
import csv
import io
import zipfile
from PIL import Image
from tqdm import tqdm


from trainers.promptsrc import PromptSRC, load_clip_to_cpu, TextEncoder

BASE_TEMPLATES = {
    'OxfordPets': 'a photo of a pet.',
    'OxfordFlowers': 'a photo of flower.',
    'FGVCAircraft': 'a photo of an aircraft.',
    'DescribableTextures': 'texture.',
    'EuroSAT': 'a centered satellite photo',
    'StanfordCars': 'a photo of a vehicle.',
    'Food101': 'a photo of a dish, a type of a food.',
    'SUN397': 'a photo of a scene.',
    'Caltech101': 'a photo of an object.',
    'UCF101': 'a photo of a human action.',
    'ImageNet': 'a photo of an object.',
    'ImageNetSketch': 'a photo of a sketch.',
    'ImageNetV2': 'a photo of an object.',
    'ImageNetA': 'a photo of an object.',
    'ImageNetR': 'a photo of an object.'
}



@TRAINER_REGISTRY.register()
class PromptSRC_NeRP(PromptSRC):
    def _norm_name(self, s):
        return s.lower().replace('_', ' ').strip()
    
    def _load_class_pairs(self, classnames, dirpath="priors", is_all=True, pair_name=None):
        dataset = self.cfg.DATASET.NAME

        if pair_name is None:
            pair_name = dataset if is_all else f"{dataset}_sub"

        bundle_path = os.path.join(dirpath, "class_pairs.json.gz")

        pairs = None
        source = None

        if os.path.isfile(bundle_path):
            with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
                self._class_pairs_bundle_cache = json.load(f)

            bundle = self._class_pairs_bundle_cache
            pairs = bundle.get(pair_name, None)
            source = f"{bundle_path}::{pair_name}"
        else:
            raise FileNotFoundError

        lut = {self._norm_name(n): i for i, n in enumerate(classnames)}

        idx_pairs = []

        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                print(f"[Pairs] Skip malformed pair: {pair}")
                continue
            a, b = pair
            ia = lut.get(self._norm_name(a), None)
            ib = lut.get(self._norm_name(b), None)
            if ia is None or ib is None:
                print(f"[Pairs] Skip unmatched pair: ({a}, {b})")
                continue
            if ia == ib:
                continue

            idx_pairs.append((ia, ib))

        self._class_pairs_idx = idx_pairs

        print(f"[Pairs] Loaded {len(idx_pairs)} pairs from {source}")
        return idx_pairs
    @torch.no_grad()
    def _get_text_prior(self, example_input):
        residual_prior  = bool(getattr(self.cfg.TEST, "RESIDUAL_TEXT_PRIOR", False))
        coe = float(getattr(self.cfg.TEST, "TEXT_PRIOR_COE", 1.))

        dataset = self.cfg.DATASET.NAME
        base_tpl = BASE_TEMPLATES[dataset]

        _, _, text_features, _ = self.model(example_input)   # [C,D]
        text_features = F.normalize(text_features, dim=1)

        clip_model = load_clip_to_cpu(self.cfg, True).to(self.device)
  
        text_encoder_clip = TextEncoder(clip_model)

        def _encode_one(s: str):
            tokens  = clip.tokenize(s).to(self.device)                                 
            tok_emb = clip_model.token_embedding(tokens).type(clip_model.dtype).to(self.device)         
            with torch.no_grad():
                t_zs = text_encoder_clip(tok_emb, tokens)  # [1, D]
                t = self.model.text_encoder(tok_emb, tokens)
            return t_zs.squeeze(0), t.squeeze(0)

        if isinstance(base_tpl, (list, tuple)):
            # FIX
            t_list = [_encode_one(str(s)) for s in base_tpl]
            t_prior = torch.stack(t_list, 0).mean(0)
        else:
            t_prior = _encode_one(str(base_tpl))[1]
            t_zs_prior = _encode_one(str(base_tpl))[0]

        
        t_prior = F.normalize(t_prior, dim=0)  # [D]
        t_zs_prior = F.normalize(t_zs_prior, dim=0)  # [D]

        b_alpha = (text_features @ t_prior)             # [C]
        b_alpha_zs = (text_features @ t_zs_prior)             # [C]

        logit_scale = self.model.logit_scale.exp()

        if residual_prior:
            self._text_prior_logits = logit_scale * (b_alpha_zs - coe * b_alpha)  
        else: 
            self._text_prior_logits = logit_scale * b_alpha_zs       # [C]



    @torch.no_grad()
    def _get_image_prior(self, example_input):
        residual_prior = bool(getattr(self.cfg.TEST, "RESIDUAL_IMAGE_PRIOR", False))
        coe = float(getattr(self.cfg.TEST, "IMAGE_PRIOR_COE", 1.))

        device = self.device if hasattr(self, "device") else example_input.device
        dtype = example_input.dtype

        dataset = self.cfg.DATASET.NAME if 'ImageNet' not in self.cfg.DATASET.NAME else 'ImageNet'
        prior_dir = getattr(self.cfg.TEST, "PRIOR_DIR", "priors")

        img_name = f"{dataset}_train_image_mean.png"
        zip_path = os.path.join(prior_dir, "train_image_mean.zip")

        img = None

        if os.path.isfile(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                if img_name in zf.namelist():
                    with zf.open(img_name, "r") as f:
                        img_bytes = f.read()

                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                else:
                    raise FileNotFoundError
        else:
            raise FileNotFoundError

        input_size = tuple(self.cfg.INPUT.SIZE)                 # e.g., (224, 224)
        resize_edge = max(input_size)                           # 224
        interp_mode = InterpolationMode.BICUBIC

        mean = self.cfg.INPUT.PIXEL_MEAN
        std  = self.cfg.INPUT.PIXEL_STD

        tfm = T.Compose([
            T.Resize(resize_edge, interpolation=interp_mode),
            T.CenterCrop((input_size[0], input_size[1])),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        x = tfm(img).unsqueeze(0).to(device=device)
        x = x.to(dtype=dtype)
        _, image_features, text_features, text_features_clip = self.model(x)  # feats: [1,D] / [C,D]

        logit_scale = self.model.logit_scale.exp()

        prior_logits = logit_scale * (image_features @ text_features.t())     # [1, C]
        prior_logits_zs = logit_scale * (image_features @ text_features_clip.t())     # [1, C]

        prior_logits_zs = prior_logits_zs.squeeze(0)                             # [C]
        prior_logits = prior_logits.squeeze(0)                             # [C]

        if residual_prior:
            self._image_prior_logits = prior_logits_zs - coe * prior_logits
        else:
            self._image_prior_logits = prior_logits_zs

    def _flip_by_pairs(self, output, pair_map=None, delta=0.6, tau=0.8):
        flip_by_prob     = bool(getattr(self.cfg.TEST, "FLIP_BY_PROB", False))
        flip_by_entropy  = bool(getattr(self.cfg.TEST, "FLIP_BY_ENTROPY", False))
        entropy_thres = float(getattr(self.cfg.TEST, "ENTROPY_THRES", 1.4))
        flip_rule = str(getattr(self.cfg.TEST, "FLIP_RULE", 'OR'))

        out = output.clone()
        top1 = out.argmax(dim=1)  # [B]
        probs_used = torch.nn.functional.softmax(output, dim=1)

        image_prior = self._image_prior_logits
        text_prior = self._text_prior_logits

        if pair_map is None:
            return out

        in_map = [int(k.item()) in pair_map for k in top1]
        rows = torch.nonzero(torch.tensor(in_map, device=out.device), as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return out

        eps = 1e-12

        for r in rows.tolist():
            i = int(top1[r].item())

            topk_prob, _ = probs_used[r].topk(5, dim=0)
            p5 = topk_prob / (topk_prob.sum() + eps)
            entropy_top5 = float((-(p5 * (p5 + eps).log())).sum().item())

            eligible = []
            for j in pair_map[i]:
                j = int(j)
                if j == i:
                    continue
                text_gap = (text_prior[i] - text_prior[j]).item()
                image_gap = (image_prior[i] - image_prior[j]).item()
                score = float(text_gap + image_gap)
                if score < tau:
                    continue

                margin = float((output[r, i] - output[r, j]).item())

                if flip_rule == 'OR':
                    if margin > delta:
                        if not flip_by_entropy and not flip_by_prob:
                            continue
                        if flip_by_entropy and (entropy_top5 < entropy_thres):
                            continue
                        if flip_by_prob:
                            if topk_prob[0].item() >= 0.6:
                                continue
                else:
                    if margin > delta:
                            continue
                    if flip_by_entropy:
                        if entropy_top5 < entropy_thres:
                            continue
                    else:
                        if flip_by_prob:
                            if topk_prob[0].item() >= 0.6:
                                continue
                eligible.append(j)

            if not eligible:
                continue

            eligible_scores = output[r, eligible]
            best_pos = int(eligible_scores.argmax().item())
            best_j = int(eligible[best_pos])

            out[r, best_j] = out[r, i] + 1e-1

        return out


    # for flip
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline + misclassification analysis."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        task = self.cfg.TASK

        classnames = list(self.dm.dataset.classnames)
        self.prior_dir = "PromptSRC_NeRP"

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)   # label: [B]
            if batch_idx == 0:
                classnames = list(self.dm.dataset.classnames)
                pair_dir = getattr(self.cfg.TEST, "PAIR_DIR", f"priors/{self.prior_dir}")
                self._load_class_pairs(classnames, dirpath=pair_dir, is_all=(task == "CD"))
                # self._load_class_pairs(classnames, dirpath=pair_dir)

                self._get_text_prior(input)
                self._get_image_prior(example_input=input)
                if getattr(self, "_class_pairs_idx", None):
                    pair_map = {}
                    for a, b in self._class_pairs_idx:
                        pair_map.setdefault(a, []).append(b)
                        pair_map.setdefault(b, []).append(a)

            logits, _, _, _ = self.model(input)

            output = logits

            classnames = list(self.dm.dataset.classnames)
            output = self._flip_by_pairs(
                output=output,
                pair_map=pair_map,
                delta=self.cfg.TEST.THRES,   
                tau=self.cfg.TEST.TAU  
            )
            self.evaluator.process(output, label)


        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
