# Demo media and publication notes

Two separate, real GazeGuard screen recordings are included in `assets/demo/`.

| Asset | Original | Processing | Output |
|---|---|---|---|
| `GazeGuard_Hand_Tracking_Full.mp4` | `Screen Recording 2026-11-19 163628.mp4` | All content retained, original 1× speed; optimized resolution/compression | ~27.5 s |
| `GazeGuard_Gaze_Detection_1_5x.mp4` | `Screen Recording 2026-11-19 163748.mp4` | All content retained, 1.5× speed; optimized resolution/compression and audio | ~17.9 s |

The original third recording (`16:28:13`) is not included in this two-video release.

- `hand_tracking_preview.jpg` shows the hand landmark overlay and links to the complete hand demo.
- `gaze_tracking_preview.jpg` links to the accelerated gaze demonstration.
- Thumbnail links work in GitHub's Markdown renderer; embedded HTML video players may be stripped and are intentionally avoided.
- The recordings are **qualitative demonstrations**, not measured accuracy, latency, or cheating-detection evaluations.
- The visible participant must consent to public publication before pushing the video files to a public repository.
- Raw Kaggle data, private landmark recordings, biometric CSVs, and local models must remain untracked.

## Upload/update commands

From the repository root, after inspecting the media and confirming permission to publish:

```powershell
git add README.md .gitignore docs/DEMO_MEDIA.md scripts/check_repo_safety.py assets/demo/
py -3.14 scripts/check_repo_safety.py --staged
git --no-pager diff --cached --stat
git commit -m "docs: add separate hand and gaze tracking demonstrations"
git push origin main
```

If your local checkout lacks `scripts/check_repo_safety.py`, use the file supplied in the release package.
