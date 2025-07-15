Before starting the project, the biggest challenge I anticipated was reliable identification of players. Since each team wears a fixed outfit, the model might easily confuse players—especially when teams wear similar colors. Plus, the moving camera makes things trickier. Identifying a player correctly before they go out of frame is a real challenge.

My first idea was to use OCR on jersey numbers, since they're unique and could be directly mapped to players. But OCR in this context isn’t very reliable. Detecting the jersey, extracting the number region, de-skewing, denoising, and then applying OCR is both resource-heavy and inaccurate—not ideal for real-time tracking.

The second idea was based on a YouTube video by Pysource that implemented Re-identification using a gallery. It’s a solid approach but a bit more involved.

Eventually, I went with DeepSORT instead of classic SORT, mainly because classic SORT tends to struggle with occlusion and overlapping players. DeepSORT, with its appearance features, performs better in crowded scenes.

After testing locally on my NVIDIA GTX 1650 Ti, the model didn't perform great in terms of FPS. For best accuracy, I found that a confidence threshold of 0.70+ works well—lower values result in false positives, especially from background objects.

The model has four classes:

0: Ball

1: Goalkeeper

2: Player

3: Referee

Since I need real-time re-identification, but my system can’t handle it well, I’ll define minimum hardware requirements instead of optimizing the model aggressively (for now).

I’m also writing to an output file frame by frame, so I need to ensure that playback maintains a fixed FPS. If processing is faster than the delay, I should still respect the original video timing—not speed it up.

In testing, the model detected 54 players, which clearly isn’t correct. I suspect duplicate boxes and false positives. Showing the detection confidence on-screen should help debug this.

Thresholds (confidence, IoU, area) definitely need fine-tuning. For example, multiple boxes on the same player can often be solved by increasing the NMS IoU threshold.

DeepSORT does help a lot with occlusion handling—but it still struggles with consistent ID assignment. Some players randomly get IDs in the 100+ range. Oddly enough, classic SORT seems more stable with ID continuity in some cases, though it lacks occlusion handling.

Finally, since processing is too slow, especially with DeepSORT on my setup, I’ll try multi-threading the DeepSORT pipeline to improve throughput.
