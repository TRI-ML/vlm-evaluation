from collections import OrderedDict


# Arrange keys in display priority order (high --> low)
MODEL_ID_TO_NAME = OrderedDict(
    [
        (
            "prism-dinosiglip+13b",
            "Prism 13B",
        ),
        (
            "prism-dinosiglip+7b",
            "Prism 7B",
        ),
        (
            "prism-dinosiglip-controlled+13b",
            "Prism 13B (Controlled)",
        ),
        (
            "prism-dinosiglip-controlled+7b",
            "Prism 7B (Controlled)",
        ),
        ("llava-v1.5-13b", "LLaVA 1.5 13B"),
        ("llava-v1.5-7b", "LLaVA 1.5 7B"),
        ("instructblip-vicuna-7b", "InstructBLIP 7B"),
    ]
)

INTERACTION_MODES_MAP = OrderedDict(
    [
        ("Chat", "chat"),
        ("Captioning", "captioning"),
        ("Bounding Box Prediction", "bbox_pred"),
        ("Visual Question Answering", "vqa"),
        ("True/False Visual Question Answering", "true_false"),
    ]
)
