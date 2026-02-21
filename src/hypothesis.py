LABELS = {

    "WORKING": {
        "triggers": {
            "hand_positions": [
                "One hand is visible holding a trowel or masonry spatula",
                "One hand is visible holding a hammer or mallet",
                "One hand is visible holding a power drill or impact driver",
                "One hand is visible holding a screwdriver",
                "One hand is visible holding a utility knife or box cutter",
                "One hand is visible holding a caulk gun",
                "One hand is visible holding a paint brush or roller",
                "One hand is visible holding a staple gun",
                "One hand is visible holding a saw",
                "One hand is visible holding a brick or concrete block",
                "Both hands are visible and gripping the same large tool together",
                "Both hands are visible and one is holding a tool while the other steadies material",
                "Both hands are visible and pressing or pushing against a surface",
                "Both hands are inside a wall cavity or enclosed space with wrists visible",
            ],
            "interaction_world": [
                "Hands and a trowel are visible pressing against a surface with wet mortar visible",
                "Hands are visible holding a brick positioned above or against a mortar surface",
                "Worker is spreading mortar or concrete mix on a surface",
                "Worker is placing or positioning a brick or block onto mortar",
                "Worker is tapping or leveling a brick or block into place",
                "Worker is hammering a nail or fastener into a surface",
                "Worker is drilling into a wall or ceiling surface",
                "Worker is applying adhesive sealant or compound to a surface",
                "Worker is fitting a component into an opening or slot",
                "Worker is fastening or tightening a connection",
            ],
        },
        "logic": "hand_positions AND interaction_world both active",
        "vlm_hypothesis": None,
    },

    "TRANSIT": {
        "triggers": {
            "hand_positions": [
                "No hands are visible in the frame",
                "Both hands are visible and both are empty",
                "Hands are visible gripping an object at chest height away from any surface",
            ],
            "interaction_world": [
                "Worker is not visibly interacting with any surface or material",
                "Hands are visible gripping a wheelbarrow or cart handle",
            ],
            "movement": [
                "Worker body is leaning at a significant forward angle suggesting forward momentum",
                "Entire frame is uniformly blurred suggesting camera was moving at moment of capture",
                "Camera view shows stair treads at an angle suggesting wearer is on a staircase",
                "Camera view is at an angle consistent with being on a ladder with rungs partially visible",
            ],
        },
        "logic": "hands empty OR carrying AND movement blur present AND no interaction",
        "vlm_hypothesis": None,
    },

    "IDLE_WAITING": {
        "triggers": {
            "hand_positions": [
                "Both hands are visible and both are empty",
                "One hand is visible and it is empty",
                "No hands are visible in the frame",
            ],
            "interaction_world": [
                "Worker is not visibly interacting with any surface or material",
            ],
            "movement": [
                "Frame is sharp with no motion blur on subject or background",
            ],
            "camera_orientation": [
                "Camera is level and pointing at an open empty space or room",
                "Camera is level and pointing at another persons face or upper body",
                "Camera is pointing at a group of two or more workers visible in the frame",
            ],
        },
        "logic": "hands empty AND no interaction AND no motion blur AND camera level",
        "vlm_hypothesis": (
            "Worker appears idle with no active task. Possible causes: waiting for material "
            "delivery, waiting for another trade to finish, unclear task assignment, or "
            "no supervisor direction. Recommend checking material staging and task scheduling."
        ),
    },

    "IDLE_DEVICE": {
        "triggers": {
            "hand_positions": [
                "One hand is visible holding a phone or tablet",
            ],
            "interaction_world": [
                "Worker is not visibly interacting with any surface or material",
            ],
            "movement": [
                "Frame is sharp with no motion blur on subject or background",
            ],
        },
        "logic": "phone in hand AND no surface interaction AND no motion",
        "vlm_hypothesis": (
            "Worker is on a phone or device for an extended period. Could indicate personal "
            "use OR worker calling supervisor because they are unsure of next task. "
            "If paired with prior IDLE_WAITING frames, likely a task clarity issue."
        ),
    },

    "WORKING_PAUSE": {
        "triggers": {
            "hand_positions": [
                "One hand is visible holding a power drill or impact driver",
                "One hand is visible holding a saw",
                "One hand is visible holding a trowel or masonry spatula",
                "One hand is visible holding a level or straight edge",
                "One hand is visible holding a measuring tape",
                "Both hands are visible and gripping the same large tool together",
                "One hand is visible holding an unidentifiable object",
                "Both hands are visible and both are empty",
                "One hand is visible and it is empty",
                "No hands are visible in the frame",
            ],
            "interaction_world": [
                "Worker is not visibly interacting with any surface or material",
                "Worker appears to be observing or inspecting without touching",
                "Hands are visible holding a level against a flat surface",
                "Worker is marking or measuring a surface",
            ],
            "movement": [
                "Frame is sharp with no motion blur on subject or background",
                "Worker body is leaning at a significant forward angle suggesting forward momentum",
                "Entire frame is uniformly blurred suggesting camera was moving at moment of capture",
            ],
            "camera_orientation": [
                "Camera is level and pointing at a wall surface at approximately chest height",
                "Camera is pointing steeply downward and the floor or ground is the main subject",
                "Camera is close range and a single hand and tool against a surface fills the frame",
                "Camera is extremely close to a surface and fills frame with texture or material detail",
                "Camera is level and pointing at an open empty space or room",
                "Camera is pointing at another workers hands or work area from close range",
            ],
        },
        "logic": (
            "Task has stalled. Covers three sub-types the VLM will distinguish: "
            "(1) TOOL_ISSUE: tool in hand, not touching surface, stationary. "
            "(2) INSPECTION_PAUSE: empty hands or measuring tool, looking at completed surface. "
            "(3) MATERIAL_SEARCH: hands empty, moving, looking at floor or empty space."
        ),
        "vlm_hypothesis": (
            "Work has paused. Determine which best explains the segment: "
            "(1) TOOL_ISSUE — worker holding tool but not applying it, possibly malfunctioning "
            "or wrong tool for the task. Check equipment maintenance log. "
            "(2) INSPECTION_PAUSE — worker checking or measuring completed work. Normal if brief, "
            "concerning if frequent. May indicate quality anxiety or unclear specs. "
            "(3) MATERIAL_SEARCH — worker moving with empty hands looking at floor or surroundings, "
            "likely cannot find required material or tool. Check staging and delivery schedule."
        ),
    },

    "UNSAFE_BEHAVIOR": {
        "triggers": {
            "hand_positions": [
                "Hands are visible and worker has bare ungloved hands",
                "Both hands are inside a wall cavity or enclosed space with wrists visible",
            ],
            "camera_orientation": [
                "Camera is pointing upward and hands are visible working on an overhead surface",
                "Camera is pointing upward and a ladder or scaffolding structure is visible above",
                "Camera is pointing steeply downward and the floor or ground is the main subject",
            ],
            "movement": [
                "Camera view is at an angle consistent with being on a ladder with rungs partially visible",
                "Camera view shows an elevated platform or scaffolding surface underfoot",
                "Worker is in a low position with hands near ground level visible in lower frame",
            ],
        },
        "logic": "no PPE visible OR elevated position with awkward reach OR hands in enclosed hazard space",
        "vlm_hypothesis": (
            "Frame suggests potential safety concern. Worker may be without required PPE, "
            "in an awkward elevated position, or reaching into an enclosed space without visible protection. "
            "Flag for safety officer review. Do not rely solely on this flag — human review required."
        ),
    },

    "BREAK": {
        "triggers": {
            "hand_positions": [
                "One hand is visible holding food or a drink",
                "Both hands are visible and both are empty",
            ],
            "interaction_world": [
                "Worker is not visibly interacting with any surface or material",
            ],
            "camera_orientation": [
                "Camera is level and pointing at an open empty space or room",
                "Camera is pointing steeply downward and the floor or ground is the main subject",
            ],
            "movement": [
                "Frame is sharp with no motion blur on subject or background",
            ],
        },
        "logic": "food or drink in hand OR hands empty AND no interaction AND stationary AND not near work surface",
        "vlm_hypothesis": None,
    },

    "UNCERTAIN": {
        "triggers": {
            "camera_state": [
                "Frame is heavily motion blurred and subject is unclear",
                "Frame is mostly dark with very low visibility",
                "Frame is mostly blocked by clothing or a body part very close to lens",
                "Frame shows heavy uniform motion blur making subject details unclear",
            ],
            "hand_positions": [
                "Not sure what the hands are doing or holding",
            ],
            "interaction_world": [
                "Not sure what the worker is interacting with",
            ],
        },
        "logic": "any critical category returns its fallback option OR camera state is obstructed or blurred",
        "vlm_hypothesis": None,
    },
}