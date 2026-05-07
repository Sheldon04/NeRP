def get_dataset_specified_config(dataset, trainer, task, subsample="new"):
    """Get dataset specific."""
    if trainer == "MMRL_NeRP":
        if subsample == "new":
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 0.1,
                    "TEST.TAU": 0.0, # -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.4,
                    "TEST.TAU": 0.3,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "SUN397": {
                    "TEST.THRES": 0.08,
                    "TEST.TAU": -1.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                },
                "DescribableTextures": {
                    "TEST.THRES": 1.5,
                    "TEST.TAU": 0.4,
                },
                "Food101": {
                    "TEST.THRES": 1.5,
                    "TEST.TAU": 1.0,
                },
                "OxfordFlowers": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                },
                "UCF101": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True
                },
                "ImageNet": {
                    "TEST.THRES": 0.05,
                    "TEST.TAU": -2.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "Caltech101": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 1.0,
                    "TEST.TAU": 0.5,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True
                },
                "EuroSAT": {
                    "TRAINER.MMRL.REP_DIM": 2048,
                    "TEST.THRES": 1.5,
                    "TEST.TAU": 1.0,
                },
            }.get(dataset, {})
        else:
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 1.0, 
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.0002,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.6,
                },
                "SUN397": {
                    "TEST.THRES": 0.14,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.44,
                    "TEST.FLIP_RULE": "AND"
                },
                "DescribableTextures": {
                    "TEST.THRES": 0.006,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.57,
                },
                "Food101": {
                    "TEST.THRES": 0.08,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.45,
                    "TEST.FLIP_RULE": "AND"
                },
                "OxfordFlowers": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "UCF101": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True
                },
                "ImageNetA": {
                    "TEST.THRES": 0.0006,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.3,
                },
                "ImageNetR": {
                    "TEST.THRES": 0.005,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.47,
                },
                "ImageNetSketch": {
                    "TEST.THRES": 0.002,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.3,
                },
                "ImageNetV2": {
                    "TEST.THRES": 0.0,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "Caltech101": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 0.004,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_RULE": "AND",
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                },
                "EuroSAT": {
                    "TEST.THRES": 1.5,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
            }.get(dataset, {})
    elif trainer == "PromptSRC_NeRP":
        if subsample == "new":
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 0.2,
                    "TEST.TAU": -2.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.004,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.56,
                },
                "SUN397": {
                    "TEST.THRES": 0.04,
                    "TEST.TAU": -2.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.34,
                    "TEST.FLIP_RULE": "AND",
                },
                "DescribableTextures": {
                    "TEST.THRES": 0.0,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.57,
                },
                "Food101": {
                    "TEST.THRES": 0.05,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.3,
                    "TEST.FLIP_RULE": "AND",
                },
                "OxfordFlowers": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                },
                "UCF101": {
                    "TEST.THRES": 0.04,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.42,
                    "TEST.FLIP_RULE": "AND",
                },
                "ImageNet": {
                    "TEST.THRES": 0.005,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "Caltech101": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": -0.5,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 0.2,
                    "TEST.TAU": -2.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 0.75,
                    "TEST.FLIP_RULE": "AND",
                },
                "EuroSAT": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU":   1.0,
                },
            }.get(dataset, {})
        else:
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 0.05, 
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.54,
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.05,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.54,
                },
                "SUN397": {
                    "TEST.THRES": 0.12,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                    "TEST.FLIP_RULE": "AND"
                },
                "DescribableTextures": {
                    "TEST.THRES": 0.006,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "Food101": {
                    "TEST.THRES": 0.25,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.45,
                    "TEST.FLIP_RULE": "AND"
                },
                "OxfordFlowers": {
                    "TEST.THRES": 3.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "UCF101": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "ImageNetA": {
                    "TEST.THRES": 0.004,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                },
                "ImageNetR": {
                    "TEST.THRES": 0.005,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.47,
                },
                "ImageNetSketch": {
                    "TEST.THRES": 0.005,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.3,
                },
                "ImageNetV2": {
                    "TEST.THRES": 0.004,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "Caltech101": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 0.2,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_RULE": "AND",
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.2,
                },
                "EuroSAT": {
                    "TEST.THRES": 1.5,
                    "TEST.TAU": 1.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
            }.get(dataset, {})
    elif trainer == "CoCoOp_NeRP":
        if subsample == "new":
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 0.15,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.47,
                    "TEST.FLIP_RULE": "AND",
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.008,
                    "TEST.TAU": -5.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.56,
                },
                "SUN397": {
                    "TEST.THRES": 0.1,
                    "TEST.TAU": -2.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.38,
                    "TEST.FLIP_RULE": "AND",
                },
                "DescribableTextures": {
                    "TEST.THRES": 0.1,
                    "TEST.TAU": -20.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.ENTROPY_THRES": 1.4,
                },
                "Food101": {
                    "TEST.THRES": 0.05,
                    "TEST.TAU": -5.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.25,
                    "TEST.FLIP_RULE": "AND",
                },
                "OxfordFlowers": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "UCF101": {
                    "TEST.THRES": 0.06,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                    "TEST.FLIP_RULE": "AND",
                },
                "ImageNet": {
                    "TEST.THRES": 0.003,
                    "TEST.TAU": -5.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                },
                "Caltech101": {
                    "TEST.THRES": 1.0,
                    "TEST.TAU": -0.5,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 1.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "EuroSAT": {
                    "TEST.THRES": 1.5,
                    "TEST.TAU":   1.0,
                },
            }.get(dataset, {})
        else:
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 1.0, 
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.03,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.58,
                    "TEST.FLIP_RULE": "AND"
                },
                "SUN397": {
                    "TEST.THRES": 0.04,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.48,
                    "TEST.FLIP_RULE": "AND"
                },
                "DescribableTextures": {
                    "TEST.THRES": 0.007,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.54,
                },
                "Food101": {
                    "TEST.THRES": 0.08,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.45,
                    "TEST.FLIP_RULE": "AND"
                },
                "OxfordFlowers": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "UCF101": {
                    "TEST.THRES": 0.1,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.44,
                    "TEST.FLIP_RULE": "AND"
                },
                "ImageNetA": {
                    "TEST.THRES": 0.0006,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.36,
                },
                "ImageNetR": {
                    "TEST.THRES": 0.006,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.47,
                },
                "ImageNetSketch": {
                    "TEST.THRES": 0.004,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.48,
                },
                "ImageNetV2": {
                    "TEST.THRES": 0.0025,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "Caltech101": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 0.36,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_RULE": "AND",
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.3,
                },
                "EuroSAT": {
                    "TEST.THRES": 1.5,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
            }.get(dataset, {})
    elif trainer == "MultiModalAdapter_NeRP":
        if subsample == "new":
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 0.001,
                    "TEST.TAU": -2.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.005,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "SUN397": {
                    "TEST.THRES": 0.1,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.33,
                    "TEST.FLIP_RULE": "AND",
                },
                "DescribableTextures": {
                    "TEST.THRES": 0.0008,
                    "TEST.TAU": -5.0,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.4,
                },
                "Food101": {
                    "TEST.THRES": 0.02,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.38,
                    "TEST.FLIP_RULE": "AND",
                },
                "OxfordFlowers": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "UCF101": {
                    "TEST.THRES": 0.18,
                    "TEST.TAU": -5.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                    "TEST.FLIP_RULE": "AND",
                },
                "ImageNet": {
                    "TEST.THRES": 0.001,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.5,
                },
                "Caltech101": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 0.02,
                    "TEST.TAU": -2.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 0.7,
                    "TEST.FLIP_RULE": "AND",
                },
                "EuroSAT": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": 0.1
                },
            }.get(dataset, {})
        else:
            cfg = {
                "StanfordCars": {
                    "TEST.THRES": 1.0, 
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                },
                "FGVCAircraft": {
                    "TEST.THRES": 0.0,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.6,
                },
                "SUN397": {
                    "TEST.THRES": 0.08,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.36,
                    "TEST.FLIP_RULE": "AND"
                },
                "DescribableTextures": {
                    "TEST.THRES": 0.0006,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.57,
                },
                "Food101": {
                    "TEST.THRES": 0.08,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.45,
                    "TEST.FLIP_RULE": "AND"
                },
                "OxfordFlowers": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
                "UCF101": {
                    "TEST.THRES": 2.0,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_BY_PROB": True
                },
                "ImageNetA": {
                    "TEST.THRES": 0.0004,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.2,
                },
                "ImageNetR": {
                    "TEST.THRES": 0.0008,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.46,
                },
                "ImageNetSketch": {
                    "TEST.THRES": 0.0006,
                    "TEST.TAU": -4.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.44,
                },
                "ImageNetV2": {
                    "TEST.THRES": 0.0,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.35,
                },
                "Caltech101": {
                    "TEST.THRES": 0.6,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.RESIDUAL_IMAGE_PRIOR": True,
                },
                "OxfordPets": {
                    "TEST.THRES": 0.4,
                    "TEST.TAU": -3.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                    "TEST.FLIP_RULE": "AND",
                    "TEST.FLIP_BY_ENTROPY": True,
                    "TEST.ENTROPY_THRES": 1.2,
                },
                "EuroSAT": {
                    "TEST.THRES": 1.5,
                    "TEST.TAU": 0.0,
                    "TEST.RESIDUAL_TEXT_PRIOR": True,
                },
            }.get(dataset, {})
    elif trainer == "MMRL":
        if subsample == "new":
            cfg = {
                "StanfordCars": {
                    "TRAINER.MMRL.REG_WEIGHT": 7.0,
                },
                "FGVCAircraft": {
                    "TRAINER.MMRL.REG_WEIGHT": 6.0,
                },
                "SUN397": {
                    "TRAINER.MMRL.REG_WEIGHT": 6.0,
                },
                "DescribableTextures": {
                    "TRAINER.MMRL.REG_WEIGHT": 6.0,
                },
                "Food101": {
                    "TRAINER.MMRL.REG_WEIGHT": 5.0,
                },
                "OxfordFlowers": {
                    "TRAINER.MMRL.REG_WEIGHT": 4.0,
                },
                "UCF101": {
                    "TRAINER.MMRL.REG_WEIGHT": 3.0,
                },
                "ImageNet": {
                    "TRAINER.MMRL.REG_WEIGHT": 0.5,
                },
                "Caltech101": {
                    "TRAINER.MMRL.REG_WEIGHT": 0.5,
                },
                "OxfordPets": {
                    "TRAINER.MMRL.REG_WEIGHT": 0.2,
                },
                "EuroSAT": {
                    "TRAINER.MMRL.REP_DIM": 2048,
                    "TRAINER.MMRL.REG_WEIGHT": 0.01
                },
            }.get(dataset, {})
        else:
            cfg = {}.get(dataset, {})
    return [item for pair in cfg.items() for item in pair]