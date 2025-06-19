### face_tracking.py
1. Detects a face using a webcam.
2. Determines the horizontal position of the face.
3. Uses a PID controller to calculate how much to move a cartoon eye (iris image).
4. Overlays the iris on an eye background image.
5. Simulates tracking the face by making the iris "look" left, right, or center.
Eventually, this can control a servo motor using an Arduino or Raspberry Pi — but that part is commented out.
