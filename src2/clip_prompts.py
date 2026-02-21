HAND_POSITIONS = [
    # Visibility states
    "No hands are visible in the frame",
    "Arms are visible but hands are cut off or hidden from view",
    "Only fingertips or partial hands are visible",

    # Single hand states
    "One hand is visible and it is empty",
    "One hand is visible and it is holding an unidentifiable object",

    # Tools - single hand
    "One hand is visible holding a trowel or masonry spatula",
    "One hand is visible holding a hammer or mallet",
    "One hand is visible holding a power drill or impact driver",
    "One hand is visible holding a screwdriver",
    "One hand is visible holding a utility knife or box cutter",
    "One hand is visible holding a measuring tape",
    "One hand is visible holding a level or straight edge",
    "One hand is visible holding pliers or wire cutters",
    "One hand is visible holding a caulk gun",
    "One hand is visible holding a paint brush or roller",
    "One hand is visible holding a staple gun",
    "One hand is visible holding a saw",

    # Materials - single hand
    "One hand is visible holding a brick or concrete block",
    "One hand is visible holding pipes or tubing",
    "One hand is visible holding electrical wires or cables",
    "One hand is visible holding a bag or container of material",
    "One hand is visible holding a sheet of drywall or board material",
    "One hand is visible holding food or a drink",
    "One hand is visible holding a phone or tablet",
    "One hand is visible holding a pen pencil or marker",
    "One hand is visible holding paper plans or a clipboard",

    # Both hands states
    "Both hands are visible and both are empty",
    "Both hands are visible and gripping the same large tool together",
    "Both hands are visible and holding opposite ends of a long object",
    "Both hands are visible and one is holding a tool while the other steadies material",
    "Both hands are visible and both are holding different tools",
    "Both hands are visible and pressing or pushing against a surface",
    "Both hands are visible and fingers are intertwined or gripping a thin object like wire or rope",
    "Both hands are inside a wall cavity or enclosed space with wrists visible",
    "Both hands are inside a bucket or container with wrists visible",

    # Glove state
    "Hands are visible and worker is wearing orange or red gloves",
    "Hands are visible and worker is wearing grey or black gloves",
    "Hands are visible and worker has bare ungloved hands",

    # Fallback
    "Not sure what the hands are doing or holding",
]

INTERACTION_WORLD = [
    # No interaction
    "Worker is not visibly interacting with any surface or material",
    "Worker appears to be observing or inspecting without touching",

    # Wall and surface work - visible from hardhat looking at work surface
    "Hands and a trowel are visible pressing against a surface with wet mortar visible",
    "Hands are visible holding a brick positioned above or against a mortar surface",
    "Hands are visible pressing a flat sheet material against a wall surface",
    "Hands are visible holding a drill against a wall or ceiling surface",
    "Hands are visible holding a level against a flat surface",
    "Worker is scraping or smoothing a wall surface with a tool",
    "Worker is hammering a nail or fastener into a surface",
    "Worker is cutting into a wall or surface",
    "Worker is applying adhesive sealant or compound to a surface",
    "Worker is painting or coating a surface",
    "Worker is marking or measuring a surface",

    # Masonry specific
    "Worker is spreading mortar or concrete mix on a surface",
    "Worker is placing or positioning a brick or block onto mortar",
    "Worker is tapping or leveling a brick or block into place",

    # Assembly and fitting
    "Worker is fitting a component into an opening or slot",
    "Worker is fastening or tightening a connection",
    "Worker is assembling two parts together",
    "Worker is connecting pipes or fittings together",
    "Worker is installing framing or structural component",

    # Lifting and carrying - visible from hardhat as hands gripping objects
    "Hands are visible gripping an object at chest height away from any surface",
    "Hands are visible holding a broom or brush with bristles near a floor surface",
    "Hands are visible holding a cloth or rag against a tool or object",
    "Hands are visible holding wire or cable near a wall opening",
    "Hands are visible gripping a wheelbarrow or cart handle",
    "Hands are visible holding a mixing tool inside a container of wet material",
    "Worker is setting down or placing an object on a visible surface",

    # Fallback
    "Not sure what the worker is interacting with",
]

CAMERA_ORIENTATION = [
    # Looking down - implies crouching, kneeling, or working on low surface
    "Camera is pointing steeply downward and the floor or ground is the main subject",
    "Camera is pointing downward and hands are visible working on a low surface",
    "Camera is pointing downward and boots or feet of the wearer are partially visible",

    # Looking straight ahead - implies standing, working on wall at chest or eye height
    "Camera is level and pointing at a wall surface at approximately chest height",
    "Camera is level and pointing at another persons face or upper body",
    "Camera is level and pointing at an open empty space or room",
    "Camera is level and pointing at a door frame window opening or structural gap",

    # Looking up - implies working overhead or on ceiling
    "Camera is pointing upward and ceiling or overhead structure is the main subject",
    "Camera is pointing upward and hands are visible working on an overhead surface",
    "Camera is pointing upward and a ladder or scaffolding structure is visible above",

    # Very close to surface - implies detailed or precise work
    "Camera is extremely close to a surface and fills frame with texture or material detail",
    "Camera is close range and a single hand and tool against a surface fills the frame",

    # Looking at other workers
    "Camera is pointing at another worker at full body distance showing their whole body",
    "Camera is pointing at another workers hands or work area from close range",
    "Camera is pointing at a group of two or more workers visible in the frame",

    # Fallback
    "Not enough visual information to determine camera orientation or direction",
]

MOVEMENT = [
    # Surface context - what surface is visible implies where worker is
    "Camera view shows flat ground or floor as the primary surface",
    "Camera view is at an angle consistent with being on a ladder with rungs partially visible",
    "Camera view shows stair treads at an angle suggesting wearer is on a staircase",
    "Ladder rungs or rails are visible in the lower portion of the frame",
    "Worker appears to be inside an elevator or small enclosed lift cabin",
    "Camera view shows an elevated platform or scaffolding surface underfoot",

    # Body mid-position clues visible from hardhat
    "Worker body is leaning at a significant forward angle suggesting forward momentum",
    "Arms are extended forward as if pushing or reaching toward something",
    "Worker is in a low position with hands near ground level visible in lower frame",

    # Motion blur - single frame observable
    "Worker limbs appear motion blurred while background is sharp",
    "Entire frame is uniformly blurred suggesting camera was moving at moment of capture",
    "Frame is sharp with no motion blur on subject or background",
    "Subject is sharp but background shows blur suggesting subject is stationary and camera moved",

    # Fallback
    "Not enough visual information to determine position or motion state",
]

CAMERA_STATE = [
    # Blur patterns - single frame observable
    "Frame is sharp and clear with no motion blur visible anywhere",
    "Frame shows light motion blur on subject edges but image is mostly readable",
    "Frame shows heavy uniform motion blur making subject details unclear",
    "Frame shows blur concentrated on moving subject while background remains sharp",
    "Frame shows blur across entire image including background and foreground equally",

    # Orientation
    "Camera is level and pointing straight ahead horizontally",
    "Camera is tilted left and horizon appears diagonal",
    "Camera is tilted right and horizon appears diagonal",
    "Camera is pointing steeply downward toward the floor",
    "Camera is pointing steeply upward toward the ceiling or sky",
    "Camera is upside down or severely rotated",

    # Obstruction and clarity
    "Frame is clear and subject is fully visible",
    "Frame is slightly blurry but subject is still identifiable",
    "Frame is heavily motion blurred and subject is unclear",
    "Frame is mostly dark with very low visibility",
    "Frame is overexposed or washed out from bright light",
    "Frame is mostly blocked by clothing or a body part very close to lens",
    "Frame has dust smoke or particles obscuring the view",
    "Frame shows only a very close up partial view of a surface or object",

    # Fisheye specific
    "Strong fisheye distortion visible with curved edges",
    "Moderate fisheye distortion visible",
    "Minimal distortion image appears relatively normal",

    # Fallback
    "Not sure about the state or orientation of the camera",
]

CATEGORY_PROMPTS: dict[str, list[str]] = {
    "hand_positions": HAND_POSITIONS,
    "interaction_world": INTERACTION_WORLD,
    "camera_orientation": CAMERA_ORIENTATION,
    "movement": MOVEMENT,
    "camera_state": CAMERA_STATE,
}