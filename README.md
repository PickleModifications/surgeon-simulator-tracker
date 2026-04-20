# surgeon-simulator-tracker

A desktop app that watches your hand through a downward-facing webcam and presses Surgeon Simulator 2013's finger keys when you curl the matching finger.

## Mapping

| Finger | Key (game) |
|--------|:---:|
| Thumb   | Space |
| Index   | R |
| Middle  | E |
| Ring    | W |
| Pinky   | A |

Supports right-hand **and** left-hand modes. MediaPipe detects the hand automatically — set the mode in the UI to match the thresholds you've calibrated for that hand.

## Requirements

- Windows 10/11
- **Python 3.11** (MediaPipe does not support 3.12+ yet). Install from https://www.python.org/downloads/ and tick *"Add python.exe to PATH"* during setup.
- A webcam, ideally mounted overhead so it looks down at the hand on a desk.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the `keyboard` hotkey registration reliably: on Windows the `keyboard` library needs admin privileges **only** for some system-wide keys; F9 works without it, but if it doesn't trigger, right-click the terminal and "Run as administrator".

## Usage

```bash
python main.py
```

1. Pick your webcam from the dropdown.
2. Pick **Right** or **Left** hand.
3. Click **Calibrate** and follow the two prompts (all fingers extended, then all fingers curled).
4. Place your hand in a neutral flat pose so no indicators are lit.
5. Click **Start** (or press **F9**) and alt-tab into Surgeon Simulator.
6. Curl a finger to press its key; un-curl to release it. Press **F9** again to disable — this safely releases any keys the app thinks it's holding.

## Project structure

```
main.py                 # entry point
requirements.txt
src/
  camera.py             # device listing + capture thread
  hand_tracker.py       # MediaPipe Hands wrapper
  gesture.py            # curl-ratio detection + debounce + calibration math
  key_sender.py         # pydirectinput press/release with state diffing
  config.py             # JSON persistence at config.json
  ui/
    main_window.py
    video_widget.py     # QLabel that renders BGR frames
    finger_panel.py     # 5 indicator dots with key labels
```

## How detection works

For the four long fingers the "curl ratio" is `distance(tip, wrist) / distance(PIP, wrist)`. When the finger is extended the tip is farther from the wrist than the middle knuckle, giving a ratio > 1.4-ish. When the finger curls, the tip comes back toward the palm and the ratio drops below ~1.0.

The thumb uses `distance(thumb_tip, index_base) / distance(thumb_base, index_base)` because the thumb hinges sideways, not forward.

Any curl ratio below the per-finger threshold means "finger down → key pressed". A short debounce (2 frames by default) prevents chatter at the threshold.

## Troubleshooting

- **No cameras listed** — unplug and replug the webcam; restart the app. Close any other app using the camera (Zoom, Teams, OBS virtual cam).
- **Indicators flicker between lit and unlit** — lower the threshold slider for that finger, or raise `debounce_frames` in `config.json`.
- **Keys don't reach the game** — Surgeon Simulator 2013 sometimes needs the tracker launched *before* the game, or run the terminal as Administrator.
- **Left/Right hand detected wrong** — MediaPipe assumes a mirrored-webcam input. If you're using a non-mirrored setup, override with the radio buttons.
