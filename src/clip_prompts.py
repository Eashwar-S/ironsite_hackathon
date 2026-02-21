HAND_POSITIONS = [
    # Visibility states
    "No hands are visible in the frame",
    "Arms are visible but hands are cut off or hidden from view",
    "Only fingertips or partial hands are visible",
    
    # Single hand states
    "Only the left hand is visible and it is empty",
    "Only the right hand is visible and it is empty",
    "Only the left hand is visible and it is holding an unidentifiable object",
    "Only the right hand is visible and it is holding an unidentifiable object",
    
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
    "Both hands are visible and appear to be tying or fastening something",
    "Both hands are inside a wall cavity or enclosed space, wrists visible",
    "Both hands are inside a bucket or container, wrists visible",
    "Both hands are raised above head level working overhead",
    
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

    # Wall and surface work
    "Worker is pressing or pushing material flat against a wall surface",
    "Worker is scraping or smoothing a wall surface",
    "Worker is drilling into a wall or ceiling surface",
    "Worker is hammering a nail or fastener into a surface",
    "Worker is cutting into a wall or surface",
    "Worker is applying adhesive sealant or compound to a surface",
    "Worker is painting or coating a surface",
    "Worker is marking or measuring a surface",

    # Masonry specific
    "Worker is spreading mortar or concrete mix on a surface",
    "Worker is placing or positioning a brick or block onto mortar",
    "Worker is mixing concrete or mortar in a container",
    "Worker is tapping or leveling a brick or block into place",

    # Assembly and fitting
    "Worker is fitting a component into an opening or slot",
    "Worker is fastening or tightening a connection",
    "Worker is assembling two parts together",
    "Worker is threading wire or cable through an opening",
    "Worker is connecting pipes or fittings together",
    "Worker is installing framing or structural component",

    # Lifting and carrying
    "Worker is lifting a brick or block off a stack",
    "Worker is lifting a bag or heavy container off the ground",
    "Worker is carrying material across the space",
    "Worker is setting down or placing an object on the floor",
    "Worker is loading or unloading a cart or wheelbarrow",

    # Cleaning and prep
    "Worker is sweeping or clearing debris from a surface",
    "Worker is cleaning a tool or piece of equipment",
    "Worker is preparing a surface before applying material",

    # Fallback
    "Not sure what the worker is interacting with",
]

HUMAN_STANCE = [
    # Upright
    "Worker is standing fully upright with weight on both feet",
    "Worker is standing upright but leaning slightly forward",
    "Worker is standing upright but leaning slightly backward",
    "Worker is standing upright and leaning against a wall for support",
    "Worker is standing on tiptoe reaching upward",
    "Worker is standing with a wide stable stance feet apart",

    # Bent and crouched
    "Worker is bending forward at the waist with back angled downward",
    "Worker is crouching with knees significantly bent but not touching floor",
    "Worker is squatting low with knees fully bent near the floor",
    "Worker is kneeling on one knee with other foot on floor",
    "Worker is kneeling on both knees on the floor",

    # Seated and low
    "Worker is sitting on the floor",
    "Worker is sitting on a bucket crate or low object",
    "Worker is lying flat on their back on the floor",
    "Worker is lying on their side on the floor",
    "Worker is lying face down on the floor",

    # Reaching and extended
    "Worker is reaching far overhead with both arms fully extended",
    "Worker is reaching sideways with one arm extended to the side",
    "Worker is twisting at the torso while lower body stays still",
    "Worker is pressing their body close to a wall with arms in front",

    # Facing direction
    "Worker is facing directly toward the camera",
    "Worker is facing directly away from the camera",
    "Worker is facing sideways perpendicular to the camera",

    # Fallback
    "Not sure what position or stance the worker is in",
]

MOVEMENT = [
    # Stationary
    "Worker is completely still and stationary",
    "Worker is stationary but making small repetitive arm or hand motions",
    "Worker is stationary but turning their head or upper body",

    # Ground level movement
    "Worker is walking slowly across flat ground",
    "Worker is walking quickly across flat ground",
    "Worker is walking while carrying something in both hands",
    "Worker is walking while carrying something over one shoulder",
    "Worker is shuffling or sidestepping a short distance",
    "Worker is turning around or pivoting in place",
    "Worker is backing up or moving in reverse",

    # Vertical movement
    "Worker is climbing up a ladder",
    "Worker is climbing down a ladder",
    "Worker is walking up stairs",
    "Worker is walking down stairs",
    "Worker is moving on an elevated scaffolding platform",
    "Worker is stepping up onto a raised surface",
    "Worker is stepping down from a raised surface",
    "Worker is in an elevator or lift moving vertically",

    # Camera implied movement
    "Camera is moving rapidly suggesting worker is walking fast or running",
    "Camera is bouncing rhythmically suggesting worker is walking at normal pace",
    "Camera is spinning or rotating suggesting worker is turning their body",
    "Camera is tilting up or down suggesting worker is changing elevation",

    # Fallback
    "Not sure whether the worker is moving or standing still",
]

CAMERA_STATE = [
    # Stability
    "Camera is completely static with no pixel movement",
    "Camera has very minor shake but is mostly stable",
    "Camera is moderately shaky with frequent small movements",
    "Camera is very shaky and unstable making image hard to read",
    "Camera is in constant smooth movement like walking",
    "Camera is in rapid jerky movement",

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