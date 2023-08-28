# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Contains information about the pascal parts categories.
http://roozbehm.info/pascal-parts/pascal-parts.html
"""


class PartCategory:
    """
    Contains information about a part category:
        - name: the machine name for this part (e.g. "lbleg")
        - desc: the human-readable description for this part (e.g. "leg")
        - orientation: human-readable orientation for this part if it
            exists (e.g. "left back")
        - many: whether there are an unspecified quantity of this part
            in an object
        - many_range: how many possible parts of this type to allow
            (default 10)
    """

    def __init__(
        self,
        name: str,
        orig_name: str = None,
        desc: str = None,
        orientation: str = "",
        many: bool = False,
        many_range: int = 10,
    ):

        self.name = name
        self.desc = desc if desc is not None else name
        self.orig_name = orig_name if orig_name is not None else self.desc
        self.orientation = orientation
        self.many = many
        self.many_range = many_range

        self.id = None  # Will be populated later

    def copy(self):
        return PartCategory(
            self.name, self.desc, self.orientation, self.many, self.many_range
        )

# --- Start pascal parts category definition ---
categories = {
    "aeroplane": [
        PartCategory("body"),
        PartCategory("stern"),
        PartCategory("lwing", orig_name="wing", desc="wing", orientation="left"),
        PartCategory("rwing", orig_name="wing", desc="wing", orientation="right"),
        PartCategory("tail"),
        PartCategory("engine", many=True),
        PartCategory("wheel", many=True),
    ],
    "bicycle": [
        PartCategory("fwheel", orig_name="wheel", desc="wheel", orientation="front"),
        PartCategory("bwheel", orig_name="wheel", desc="wheel", orientation="back"),
        PartCategory("saddle", orig_name="seat", desc="seat"),
        PartCategory("handlebar"),
        PartCategory("chainwheel"),
        PartCategory("headlight", many=True),
    ],
    "bird": [
        PartCategory("head"),
        PartCategory("leye", orig_name="eye", desc="eye", orientation="left"),
        PartCategory("reye", orig_name="eye", desc="eye", orientation="right"),
        PartCategory("beak"),
        PartCategory("torso"),
        PartCategory("neck"),
        PartCategory("lwing", orig_name="wing", desc="wing", orientation="left"),
        PartCategory("rwing", orig_name="wing", desc="wing", orientation="right"),
        PartCategory("lleg", orig_name="leg", desc="leg", orientation="left"),
        PartCategory("lfoot", orig_name="foot", desc="foot", orientation="left"),
        PartCategory("rleg", orig_name="leg", desc="leg", orientation="right"),
        PartCategory("rfoot", orig_name="foot", desc="foot", orientation="right"),
        PartCategory("tail"),
    ],
    "boat": [],
    "bottle": [PartCategory("cap"), PartCategory("body")],
    "bus": [
        PartCategory("frontside", desc="front"),
        PartCategory("leftside", orig_name="side", desc="left"),
        PartCategory("rightside", orig_name="side", desc="right"),
        PartCategory("backside", desc="back"),
        PartCategory("roofside", orig_name="roof", desc="roof"),
        PartCategory(
            "leftmirror", orig_name="mirror", desc="mirror", orientation="left"
        ),
        PartCategory(
            "rightmirror",
            orig_name="mirror",
            desc="mirror",
            orientation="right",
        ),
        PartCategory(
            "fliplate",
            orig_name="license plate",
            desc="license plate",
            orientation="front",
        ),
        PartCategory(
            "bliplate",
            orig_name="license plate",
            desc="license plate",
            orientation="back",
        ),
        PartCategory("door", many=True),
        PartCategory("wheel", many=True),
        PartCategory("headlight", many=True),
        PartCategory("window", many=True, many_range=20),
    ],
    # Same as bus. Will be populated below.
    "car": None,
    "cat": [
        PartCategory("head"),
        PartCategory("leye", desc="eye", orientation="left"),
        PartCategory("reye", desc="eye", orientation="right"),
        PartCategory("lear", desc="ear", orientation="left"),
        PartCategory("rear", desc="ear", orientation="right"),
        PartCategory("nose"),
        PartCategory("torso"),
        PartCategory("neck"),
        PartCategory("lfleg", desc="leg", orientation="left front"),
        PartCategory("lfpa", desc="paw", orientation="left front"),
        PartCategory("rfleg", desc="leg", orientation="right front"),
        PartCategory("rfpa", desc="paw", orientation="right front"),
        PartCategory("lbleg", desc="leg", orientation="left back"),
        PartCategory("lbpa", desc="paw", orientation="left back"),
        PartCategory("rbleg", desc="leg", orientation="right back"),
        PartCategory("rbpa", desc="paw", orientation="right back"),
        PartCategory("tail"),
    ],
    "chair": [],
    "cow": [
        PartCategory("head"),
        PartCategory("leye", desc="eye", orientation="left"),
        PartCategory("reye", desc="eye", orientation="right"),
        PartCategory("lear", desc="ear", orientation="left"),
        PartCategory("rear", desc="ear", orientation="right"),
        PartCategory("muzzle"),
        PartCategory("lhorn", desc="horn", orientation="left"),
        PartCategory("rhorn", desc="horn", orientation="right"),
        PartCategory("torso"),
        PartCategory("neck"),
        PartCategory("lfuleg", desc="leg", orientation="left front upper"),
        PartCategory("lflleg", desc="leg", orientation="left front lower"),
        PartCategory("rfuleg", desc="leg", orientation="right front upper"),
        PartCategory("rflleg", desc="leg", orientation="right front lower"),
        PartCategory("lbuleg", desc="leg", orientation="left back upper"),
        PartCategory("lblleg", desc="leg", orientation="left back lower"),
        PartCategory("rbuleg", desc="leg", orientation="right back upper"),
        PartCategory("rblleg", desc="leg", orientation="right back lower"),
        PartCategory("tail"),
    ],
    "diningtable": [],
    # Dog is cat + muzzle. It'll be populated below.
    "dog": None,
    # Horse is cow + hoof.
    "horse": None,
    "motorbike": [
        PartCategory("fwheel", desc="wheel", orientation="front"),
        PartCategory("bwheel", desc="wheel", orientation="back"),
        PartCategory("handlebar"),
        PartCategory("saddle", desc="seat"),
        PartCategory("headlight", many=True),
    ],
    "person": [
        PartCategory("head"),
        PartCategory("leye", desc="eye", orientation="left"),
        PartCategory("reye", desc="eye", orientation="right"),
        PartCategory("lear", desc="ear", orientation="left"),
        PartCategory("rear", desc="ear", orientation="right"),
        PartCategory("lebrow", desc="eyebrow", orientation="left"),
        PartCategory("rebrow", desc="eyebrow", orientation="right"),
        PartCategory("nose"),
        PartCategory("mouth"),
        PartCategory("hair"),
        PartCategory("torso"),
        PartCategory("neck"),
        PartCategory("llarm", desc="arm", orientation="left lower"),
        PartCategory("luarm", desc="arm", orientation="left upper"),
        PartCategory("lhand", desc="hand", orientation="left"),
        PartCategory("rlarm", desc="arm", orientation="right lower"),
        PartCategory("ruarm", desc="arm", orientation="right upper"),
        PartCategory("rhand", desc="hand", orientation="right"),
        PartCategory("llleg", desc="leg", orientation="left lower"),
        PartCategory("luleg", desc="leg", orientation="left upper"),
        PartCategory("lfoot", desc="foot", orientation="left"),
        PartCategory("rlleg", desc="leg", orientation="right lower"),
        PartCategory("ruleg", desc="leg", orientation="right upper"),
        PartCategory("rfoot", desc="foot", orientation="right"),
    ],
    "pottedplant": [
        PartCategory("pot"),
        PartCategory("plant"),
    ],
    # Same parts as cow.
    "sheep": None,
    "sofa": [],
    "train": [
        PartCategory("head", desc="locomotive"),
        PartCategory(
            "hfrontside",
            orig_name="locomotive_front",
            desc="front of the locomotive",
        ),
        PartCategory(
            "hleftside",
            orig_name="locomotive_side",
            desc="left side of the locomotive",
        ),
        PartCategory(
            "hrightside",
            orig_name="locomotive_side",
            desc="right side of the locomotive",
        ),
        PartCategory(
            "hbackside",
            orig_name="locomotive_backside",
            desc="back of the locomotive",
        ),
        PartCategory(
            "hroofside",
            orig_name="locomotive_roof",
            desc="roof of the locomotive",
        ),
        PartCategory("headlight", many=True),
        PartCategory("coach", many=True, orig_name="coach_car", desc="coach car"),
        PartCategory(
            "cfrontside",
            many=True,
            orig_name="coach_front",
            desc="front of a coach car",
        ),
        PartCategory(
            "cleftside",
            many=True,
            orig_name="coach_side",
            desc="left side of a coach car",
        ),
        PartCategory(
            "crightside",
            many=True,
            orig_name="coach_side",
            desc="right side of a coach car",
        ),
        PartCategory(
            "cbackside",
            many=True,
            orig_name="coach_backside",
            desc="back of a coach car",
        ),
        PartCategory(
            "croofside",
            many=True,
            orig_name="coach_roof",
            desc="roof of a coach car",
        ),
    ],
    "tvmonitor": [
        PartCategory("screen"),
    ],
}


categories["car"] = categories["bus"].copy()
categories["dog"] = categories["cat"] + [PartCategory("muzzle")]
categories["horse"] = categories["cow"] + [
    PartCategory("lfho", desc="hoof", orientation="left front"),
    PartCategory("rfho", desc="hoof", orientation="right front"),
    PartCategory("lbho", desc="hoof", orientation="left back"),
    PartCategory("rbho", desc="hoof", orientation="right back"),
]
categories["sheep"] = categories["cow"].copy()

def get_orig_part(cat: str, part_name: str) -> str:
    parts = categories[cat]
    for part in parts:
        if part.name == part_name:
            return part.orig_name
