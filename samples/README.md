# Samples

Drop short recorded street clips here for offline analysis on the dev box.

Video files in this folder are **git-ignored** (see `.gitignore`) — only this
README and `.gitkeep` are tracked, so large clips never get committed.

## Workflow

1. Record 5–10 minutes from the camera on the mini-PC.
2. Copy the file here, e.g. `samples/street-test.mp4`.
3. Run the analyzer:

   ```bash
   traffic-log test --source samples/street-test.mp4 --config config/config.dev.yaml
   ```

The Docker `analyze-gpu` service expects `samples/street-test.mp4` by default.
