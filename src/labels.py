from __future__ import annotations

from typing import Literal

LABELS = {
    "WORKING": {
        "hand_positions": [
            "Hands are actively engaged with tools, controls, or material handling.",
            "Frequent purposeful hand-object interaction indicates productive task execution.",
        ],
        "interaction_world": [
            "Visible interaction with construction equipment, tools, fixtures, or work materials.",
            "Scene context supports active task progression rather than waiting.",
        ],
        "movement": [
            "Body/camera motion is consistent with active job execution rather than relocation.",
        ],
        "camera_orientation": [
            "View is mostly oriented toward the work surface, tool zone, or task target.",
        ],
    },
    "TRANSIT": {
        "hand_positions": [
            "Hands are not primarily manipulating tools; may be free, carrying, or swinging while moving.",
        ],
        "interaction_world": [
            "Scene changes indicate relocation between work areas rather than task completion.",
        ],
        "movement": [
            "Sustained locomotion (walking/climbing/relocating) dominates the frame context.",
        ],
        "camera_orientation": [
            "Forward navigation perspective with changing surroundings and path traversal cues.",
        ],
    },
    "IDLE_WAITING": {
        "hand_positions": [
            "Hands are largely inactive, resting, or not engaged in productive manipulation.",
        ],
        "interaction_world": [
            "No meaningful tool/material interaction; context suggests waiting for next dependency.",
        ],
        "movement": [
            "Minimal movement beyond small posture shifts.",
        ],
        "camera_orientation": [
            "View may linger on static scene elements without task progression.",
        ],
    },
    "IDLE_DEVICE": {
        "hand_positions": [
            "Hands appear occupied with a phone/device rather than construction work tools.",
        ],
        "interaction_world": [
            "Attention is directed to a handheld screen/device, not the immediate task environment.",
        ],
        "movement": [
            "Low movement or stationary posture while interacting with device.",
        ],
        "camera_orientation": [
            "Camera may tilt toward chest/handheld area where device is visible.",
        ],
    },
    "WORKING_PAUSE": {
        "hand_positions": [
            "Hands are near work context but temporarily inactive between active work bouts.",
        ],
        "interaction_world": [
            "Worker remains in task zone with tools/materials present but momentarily paused.",
        ],
        "movement": [
            "Short, low-motion pause interrupts otherwise active work sequence.",
        ],
        "camera_orientation": [
            "Orientation remains near the work area indicating likely imminent task continuation.",
        ],
    },
    "UNSAFE_BEHAVIOR": {
        "hand_positions": [
            "Hand actions appear risky, improper, or inconsistent with safe operating posture.",
        ],
        "interaction_world": [
            "Visible context suggests potential safety non-compliance or hazardous interaction.",
        ],
        "movement": [
            "Movement pattern suggests instability, unsafe traversal, or abrupt hazardous handling.",
        ],
        "camera_orientation": [
            "Orientation captures potentially dangerous setup, proximity, or unsafe action cues.",
        ],
    },
    "BREAK": {
        "hand_positions": [
            "Hands are disengaged from work tasks in a sustained rest posture.",
        ],
        "interaction_world": [
            "Scene suggests non-work interval (rest area, hydration, off-task pause).",
        ],
        "movement": [
            "Movement is limited and non-productive, consistent with a break period.",
        ],
        "camera_orientation": [
            "View is not centered on active task surfaces or work targets.",
        ],
    },
    "UNCERTAIN": {
        "hand_positions": [
            "Hand cues are obscured, out of frame, or insufficient for reliable labeling.",
        ],
        "interaction_world": [
            "Scene context is ambiguous, occluded, or too limited for confident interpretation.",
        ],
        "movement": [
            "Motion cues are inconclusive for assigning a specific refined label.",
        ],
        "camera_orientation": [
            "Orientation/visibility prevents grounded classification.",
        ],
    },
}

CoarseLabel = Literal["IDLE", "WORKING", "TRANSIT"]
RefinedLabel = Literal[
    "WORKING",
    "TRANSIT",
    "IDLE_WAITING",
    "IDLE_DEVICE",
    "WORKING_PAUSE",
    "UNSAFE_BEHAVIOR",
    "BREAK",
    "UNCERTAIN",
]
