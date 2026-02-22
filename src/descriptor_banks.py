"""First-person helmet-cam descriptor banks for zero-shot classification.

Each bank captures one visual dimension of what the camera wearer SEES.
Important: this is a first-person helmet-cam — the wearer's own body,
hands, and stance are NOT visible. Descriptors describe the SCENE ahead.

Every descriptor maps to a label-weight vector [IDLE, WORKING, TRANSIT, DOWNTIME]
used by ``multi_axis_classify`` in ``model.py``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
#  Label indices (must stay in sync with model.LABELS)
# ---------------------------------------------------------------------------
_I, _W, _T, _D = 0, 1, 2, 3  # IDLE, WORKING, TRANSIT, DOWNTIME

# ---------------------------------------------------------------------------
#  Bank 1 — SCENE_ACTIVITY: what activity is visible in the scene
# ---------------------------------------------------------------------------
SCENE_ACTIVITY = [
    # Active construction work visible (active movement)
    "A person's hands actively plastering or rendering a wall with wet mortar",
    "A person's hands actively laying bricks or blocks with mortar",
    "A person's hands actively drilling into a surface with visible dust",
    "A person's hands actively driving nails or screws into wood or drywall",
    "A person's hands actively placing or aligning tiles on a surface",
    "A person's hands actively fitting pipes or conduits together",
    "A person's hands actively connecting or routing electrical wires",
    "A person's hands actively applying paint or sealant to a surface",
    "A person's hands actively pouring or spreading concrete on the ground",
    "A person's hands actively cutting or sawing wood",
    "A person's hands actively bending or tying rebar or metal",
    "A person's hands actively sanding or scraping a surface smooth",
    "A person's hands actively placing insulation material into a wall cavity",
    "A person's hands using a measuring tape or level against a surface",
    # Prep and support work (active movement)
    "A person's hands actively sorting or organizing materials on a ground or table",
    "A person's hands actively mixing a bucket of mortar or plaster with a tool",
    "A person's hands actively arranging or setting up tools for a task",
    "A person's hands actively cleaning or sweeping a work area of debris",
    # ── Transit / walking scenes ──
    "A hallway or corridor stretching ahead as someone walks through it",
    "An outdoor construction site path being walked along",
    "The view of a concrete floor moving below as someone walks forward",
    "A distant work area visible as the camera moves toward it",
    "Walking past scaffolding or construction barriers on a site path",
    # No productive activity (Static/Idle)
    "A static view of a wall or floor with no movement or activity happening",
    "An empty room with no people or work being performed",
    "Looking out a window or at the sky with no activity",
    "A view of a ceiling or corner with no movement",
    "Workers standing around talking in a group without working",
    "A phone or tablet screen visible in a resting position",
    "Food, drink, or lunch items visible on a table while resting",
    "A worker sitting or resting with no task in progress",
]

SCENE_ACTIVITY_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    # (IDLE, WORKING, TRANSIT, DOWNTIME)
    # Active work → strong WORKING
    "A person's hands actively plastering or rendering a wall with wet mortar": (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively laying bricks or blocks with mortar":   (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively drilling into a surface with visible dust":     (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively driving nails or screws into wood or drywall":  (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively placing or aligning tiles on a surface":         (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively fitting pipes or conduits together":            (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively connecting or routing electrical wires":         (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively applying paint or sealant to a surface":        (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively pouring or spreading concrete on the ground":      (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively cutting or sawing wood":                            (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively bending or tying rebar or metal":                  (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively sanding or scraping a surface smooth":           (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively placing insulation material into a wall cavity":(0.0, 1.0, 0.0, 0.0),
    "A person's hands using a measuring tape or level against a surface":        (0.0, 0.9, 0.0, 0.1),
    # Prep work → mostly WORKING
    "A person's hands actively sorting or organizing materials on a ground or table": (0.1, 0.7, 0.0, 0.2),
    "A person's hands actively mixing a bucket of mortar or plaster with a tool":     (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively arranging or setting up tools for a task":              (0.1, 0.7, 0.0, 0.2),
    "A person's hands actively cleaning or sweeping a work area of debris":             (0.0, 0.8, 0.0, 0.2),
    # Transit scenes → pure TRANSIT
    "A hallway or corridor stretching ahead as someone walks through it":  (0.0, 0.0, 1.0, 0.0),
    "An outdoor construction site path being walked along":                (0.0, 0.0, 1.0, 0.0),
    "The view of a concrete floor moving below as someone walks forward":  (0.0, 0.0, 1.0, 0.0),
    "A distant work area visible as the camera moves toward it":           (0.0, 0.0, 1.0, 0.0),
    "Walking past scaffolding or construction barriers on a site path":    (0.0, 0.0, 1.0, 0.0),
    # No activity → IDLE or DOWNTIME
    "A static view of a wall or floor with no movement or activity happening": (0.8, 0.0, 0.0, 0.2),
    "An empty room with no people or work being performed":                (0.8, 0.0, 0.0, 0.2),
    "Looking out a window or at the sky with no activity":                  (0.8, 0.0, 0.0, 0.2),
    "A view of a ceiling or corner with no movement":                       (0.8, 0.0, 0.0, 0.2),
    "Workers standing around talking in a group without working":           (0.7, 0.0, 0.0, 0.3),
    "A phone or tablet screen visible in a resting position":               (0.0, 0.0, 0.0, 1.0),
    "Food, drink, or lunch items visible on a table while resting":         (0.0, 0.0, 0.0, 1.0),
    "A worker sitting or resting with no task in progress":                (0.0, 0.0, 0.0, 1.0),
}

# ---------------------------------------------------------------------------
#  Bank 2 — OBJECTS_VISIBLE: construction materials and tools in view
# ---------------------------------------------------------------------------
OBJECTS_VISIBLE = [
    # Active tools in use (active motion)
    "A person's hands actively using a trowel or float",
    "A person's hands actively using a power drill or driver",
    "A person's hands actively using a hammer or mallet",
    "A person's hands actively using a saw or cutter",
    "A person's hands actively using a paint roller or brush",
    "A person's hands actively using a caulk gun",
    "A person's hands actively using a welding torch",
    # Tools present but not in active use (Idle/Wait)
    "Construction tools lying on the floor or a table unused",
    "A toolbox or tool belt sitting unmoving on a surface",
    # Construction materials in view
    "Stacks of bricks or concrete blocks visible in a static area",
    "Bags of cement or plaster stacked and unmoving",
    "Lengths of pipe or conduit sitting on the ground",
    "Piles of lumber or timber in a storage position",
    "Rolls of wire or cable visible but not being handled",
    "Sheets of drywall or plywood leaning against a wall",
    "Scaffolding or temporary support structure in a static state",
    # Transit / movement indicators
    "A construction vehicle like a forklift or excavator moving in distance",
    "Site signage or hazard tape visible while walking",
    "The ground surface of an access road visible below while moving",
    # Non-work items
    "No tools or construction materials visible in the view",
    "Personal items like bags or clothing visible while resting",
    "A water bottle or thermos sitting while on a break",
]

OBJECTS_VISIBLE_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    # Active tools → strong WORKING
    "A person's hands actively using a trowel or float":                    (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively using a power drill or driver":            (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively using a hammer or mallet":                  (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively using a saw or cutter":                      (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively using a paint roller or brush":             (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively using a caulk gun":                        (0.0, 1.0, 0.0, 0.0),
    "A person's hands actively using a welding torch":                         (0.0, 1.0, 0.0, 0.0),
    # Tools present but idle → strong IDLE
    "Construction tools lying on the floor or a table unused":             (0.8, 0.0, 0.0, 0.2),
    "A toolbox or tool belt sitting unmoving on a surface":                (0.8, 0.0, 0.0, 0.2),
    # Materials → IDLE lean (if no action is being taken)
    "Stacks of bricks or concrete blocks visible in a static area":       (0.7, 0.1, 0.1, 0.1),
    "Bags of cement or plaster stacked and unmoving":                     (0.7, 0.1, 0.1, 0.1),
    "Lengths of pipe or conduit sitting on the ground":                  (0.7, 0.1, 0.1, 0.1),
    "Piles of lumber or timber in a storage position":                   (0.7, 0.1, 0.1, 0.1),
    "Rolls of wire or cable visible but not being handled":                (0.7, 0.1, 0.1, 0.1),
    "Sheets of drywall or plywood leaning against a wall":                 (0.7, 0.1, 0.1, 0.1),
    "Scaffolding or temporary support structure in a static state":         (0.6, 0.1, 0.2, 0.1),
    # Transit / movement indicators
    "A construction vehicle like a forklift or excavator moving in distance": (0.0, 0.0, 1.0, 0.0),
    "Site signage or hazard tape visible while walking":                 (0.0, 0.0, 1.0, 0.0),
    "The ground surface of an access road visible below while moving":     (0.0, 0.0, 1.0, 0.0),
    # Non-work → IDLE / DOWNTIME
    "No tools or construction materials visible in the view":              (0.6, 0.0, 0.2, 0.2),
    "Personal items like bags or clothing visible while resting":           (0.0, 0.0, 0.0, 1.0),
    "A water bottle or thermos sitting while on a break":                   (0.0, 0.0, 0.0, 1.0),
}

# ---------------------------------------------------------------------------
#  Bank 3 — ENVIRONMENT: the surroundings visible in the frame
# ---------------------------------------------------------------------------
ENVIRONMENT = [
    # Indoor construction zones
    "Inside a building under construction with exposed concrete walls",
    "Inside a room with partially finished drywall or plaster",
    "Inside a narrow corridor or hallway under construction",
    "Inside a building with visible electrical conduits and junction boxes",
    "Inside a building with exposed plumbing or pipe runs",
    "Inside a stairwell under construction",
    # Outdoor construction zones
    "An outdoor construction site with exposed earth or foundation trenches",
    "An outdoor site with a partially built structure of concrete or steel",
    "An outdoor paved area with construction barriers or cones visible",
    # Transition and movement areas
    "A long empty corridor or hallway being walked through",
    "An open site road or path between buildings",
    "A parking lot or staging area with vehicles and material piles",
    "An entrance or exit doorway of a building",
    "A temporary fence or gate at a construction site perimeter",
    # Rest and break areas
    "A break area or canteen with tables and chairs visible",
    "A portable toilet or restroom facility",
    "A shaded rest area with benches or seating",
    "An office trailer or portable cabin interior",
]

ENVIRONMENT_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    # Indoor work zones → slight WORKING lean
    "Inside a building under construction with exposed concrete walls":     (0.2, 0.5, 0.1, 0.2),
    "Inside a room with partially finished drywall or plaster":            (0.2, 0.5, 0.1, 0.2),
    "Inside a narrow corridor or hallway under construction":              (0.2, 0.3, 0.3, 0.2),
    "Inside a building with visible electrical conduits and junction boxes":(0.2, 0.5, 0.1, 0.2),
    "Inside a building with exposed plumbing or pipe runs":                (0.2, 0.5, 0.1, 0.2),
    "Inside a stairwell under construction":                               (0.1, 0.2, 0.6, 0.1),
    # Outdoor work zones
    "An outdoor construction site with exposed earth or foundation trenches":(0.2, 0.4, 0.2, 0.2),
    "An outdoor site with a partially built structure of concrete or steel":(0.2, 0.4, 0.2, 0.2),
    "An outdoor paved area with construction barriers or cones visible":   (0.2, 0.2, 0.4, 0.2),
    # Movement areas → TRANSIT
    "A long empty corridor or hallway being walked through":               (0.0, 0.0, 1.0, 0.0),
    "An open site road or path between buildings":                         (0.0, 0.0, 1.0, 0.0),
    "A parking lot or staging area with vehicles and material piles":      (0.1, 0.0, 0.8, 0.1),
    "An entrance or exit doorway of a building":                           (0.0, 0.0, 0.9, 0.1),
    "A temporary fence or gate at a construction site perimeter":          (0.1, 0.0, 0.8, 0.1),
    # Rest areas → DOWNTIME
    "A break area or canteen with tables and chairs visible":              (0.0, 0.0, 0.0, 1.0),
    "A portable toilet or restroom facility":                              (0.0, 0.0, 0.0, 1.0),
    "A shaded rest area with benches or seating":                          (0.0, 0.0, 0.0, 1.0),
    "An office trailer or portable cabin interior":                        (0.2, 0.0, 0.0, 0.8),
}

# ---------------------------------------------------------------------------
#  Bank 4 — MOTION_CUES: camera movement patterns from helmet-cam
# ---------------------------------------------------------------------------
MOTION_CUES = [
    # Stationary camera
    "The scene is completely still with no camera movement",
    "The scene is nearly still with only tiny vibrations from breathing",
    # Focused work movement
    "The camera is making small steady back-and-forth motions focused on one spot",
    "The camera is slowly panning across a work surface being inspected",
    "The camera angle is looking straight down at a close work surface",
    "The camera angle is tilted upward looking at work on a ceiling or high wall",
    # Walking / transit movement
    "The scene is bouncing rhythmically as if the wearer is walking",
    "The scene is moving forward quickly through a corridor or open area",
    "The scene is panning from side to side as the wearer looks around while moving",
    "The scene shows a staircase with the view bobbing up or down steps",
    # Idle patterns
    "The camera is slowly looking around the room without focusing on anything",
    "The camera is pointed at the ground or floor with no movement",
    "The camera is pointed at a distant scene with no interaction",
]

MOTION_CUES_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    # Stationary
    "The scene is completely still with no camera movement":                (0.7, 0.0, 0.0, 0.3),
    "The scene is nearly still with only tiny vibrations from breathing":   (0.7, 0.0, 0.0, 0.3),
    # Work movement → WORKING
    "The camera is making small steady back-and-forth motions focused on one spot": (0.0, 0.9, 0.0, 0.1),
    "The camera is slowly panning across a work surface being inspected":  (0.1, 0.7, 0.0, 0.2),
    "The camera angle is looking straight down at a close work surface":   (0.0, 0.9, 0.0, 0.1),
    "The camera angle is tilted upward looking at work on a ceiling or high wall": (0.0, 0.9, 0.0, 0.1),
    # Walking → TRANSIT
    "The scene is bouncing rhythmically as if the wearer is walking":      (0.0, 0.0, 1.0, 0.0),
    "The scene is moving forward quickly through a corridor or open area": (0.0, 0.0, 1.0, 0.0),
    "The scene is panning from side to side as the wearer looks around while moving": (0.0, 0.0, 0.9, 0.1),
    "The scene shows a staircase with the view bobbing up or down steps":  (0.0, 0.0, 1.0, 0.0),
    # Idle → IDLE
    "The camera is slowly looking around the room without focusing on anything": (0.7, 0.0, 0.1, 0.2),
    "The camera is pointed at the ground or floor with no movement":       (0.6, 0.0, 0.0, 0.4),
    "The camera is pointed at a distant scene with no interaction":        (0.6, 0.0, 0.2, 0.2),
}

# ---------------------------------------------------------------------------
#  Bank 5 — CAMERA_STATE  (confidence modifier, not direct label evidence)
# ---------------------------------------------------------------------------
CAMERA_STATE = [
    # Clarity
    "Frame is clear and the scene is fully visible",
    "Frame is slightly blurry but the scene is still recognizable",
    "Frame is heavily blurred and the scene is hard to identify",
    "Frame is very dark with almost no visible detail",
    "Frame is overexposed and washed out from bright light",
    # Obstructions
    "Frame is mostly blocked by something very close to the lens",
    "Frame has dust or particles floating in the air obscuring the view",
    # Stability
    "The image is sharp and stable with no motion blur",
    "The image has slight motion blur from moderate movement",
    "The image has severe motion blur making everything streaky",
]

# Confidence multiplier: 1.0 = clear/usable, < 1.0 = degrade confidence
CAMERA_STATE_CONFIDENCE: dict[str, float] = {
    "Frame is clear and the scene is fully visible":                       1.0,
    "Frame is slightly blurry but the scene is still recognizable":        0.9,
    "Frame is heavily blurred and the scene is hard to identify":          0.5,
    "Frame is very dark with almost no visible detail":                    0.4,
    "Frame is overexposed and washed out from bright light":               0.5,
    "Frame is mostly blocked by something very close to the lens":         0.3,
    "Frame has dust or particles floating in the air obscuring the view":  0.6,
    "The image is sharp and stable with no motion blur":                   1.0,
    "The image has slight motion blur from moderate movement":             0.85,
    "The image has severe motion blur making everything streaky":          0.4,
}

# ---------------------------------------------------------------------------
#  Aggregate bank config — used by pipeline / model to iterate banks
# ---------------------------------------------------------------------------
BANKS = [
    {
        "name": "scene_activity",
        "descriptors": SCENE_ACTIVITY,
        "weights": SCENE_ACTIVITY_WEIGHTS,
        "bank_weight": 1.8,   
    },
    {
        "name": "objects_visible",
        "descriptors": OBJECTS_VISIBLE,
        "weights": OBJECTS_VISIBLE_WEIGHTS,
        "bank_weight": 1.0,
    },
    {
        "name": "environment",
        "descriptors": ENVIRONMENT,
        "weights": ENVIRONMENT_WEIGHTS,
        "bank_weight": 1.0,
    },
    {
        "name": "motion_cues",
        "descriptors": MOTION_CUES,
        "weights": MOTION_CUES_WEIGHTS,
        "bank_weight": 0.5,   # Prioritize actual movement/motion over static visual cues
    },
]

# Camera state is handled separately as a confidence modifier
CAMERA_BANK = {
    "name": "camera_state",
    "descriptors": CAMERA_STATE,
    "confidence": CAMERA_STATE_CONFIDENCE,
}
