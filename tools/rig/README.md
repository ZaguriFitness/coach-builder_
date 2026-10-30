# Anatomy model rig

Builds `model/` (the skinned anatomy model used by the 3D viewer) from the source FBX.

1. `python3 extract.py Anatomy_Male_LP_v001.fbx` – reads the binary FBX, saves `mesh.npz` and the embedded textures.
2. `python3 build.py ../../model` – skeleton (spine, neck, head, arms, legs, one finger joint per hand at the
   knuckles so the hand can close around a bar or ring; no separate fingers, thumb or eyes),
   distance-based skin weights limited to each limb's own side, Laplacian weight smoothing,
   muscle segmentation from the color map (each muscle color region → a muscle group),
   and export to `anatomy.bin` / `anatomy.json` plus the color, normal and group maps.

Needs numpy, scipy and Pillow. The FBX itself is not kept in the repo.

3. `python3 texture.py ../../model/anatomy_color.jpg ../../model/anatomy_real.jpg` – the anatomy-atlas surface
   used by CoachBuilder's viewer: red muscles with a shade per muscle, darker grooves where muscles meet, fibre
   grain, ivory tendons and fascia (from the muscle-ID colour map; numpy + Pillow only).
