# Pocket cartridges

**The guest draws the pixels. Change one input, and replay the consequences.**

- [Passage Relay](../pocket/playing.gif) and its [counterfactual](../pocket/edit-vs-omitted.gif)
- [Rail Crate](rail-crate.gif)
- [Machine Monitor](monitor.gif)

[Start Pocket or write a cartridge](../../POCKET-CARTRIDGE.md). The same
manifest-driven player serves all of the cartridges. The GIFs were captured
from real browser interaction and slowed so the individual frames are
readable. Browser playback and saved VisuaML verification are labeled in
the inspector.

Rail Crate uses a five-cell rail. The player starts on cell two and the
crate on cell three. Two right inputs push it onto cell five. The next
right input is a blocked push and leaves both positions unchanged.
Replacing the first right input with Left leaves the crate on cell four
at completion. An independent model checks every position, and the
pixel checks distinguish empty cells, player, crate, rails and goal.

The Machine Monitor is a 20-column, 18-line guest terminal with original
8 × 8 glyph cells, an underscore cursor, line editing, and scrolling.
It accepts printable ASCII, Backspace, and Enter. The line accepts up to
18 characters alongside the prompt and cursor. Commands are case
insensitive. HELP lists commands, STATUS prints the tick and both agent
positions, MAP prints the passage, DIG opens it, STEP advances the world,
and RESET restores the world while retaining the transcript. Position
coordinates are zero-based. Tick advances on STEP and saturates at 999.
A guest command cannot load programs or access a filesystem.

The [edited key log](../../fixtures/monitor-play.json) finishes with
`MAP AB.`. The [counterfactual](../../fixtures/monitor-omitted.json)
omits the four DIG key events and finishes with `MAP A#B`. The regression
log exercises Backspace, HELP, an unknown command, RESET and scroll.
Both typed histories match independent transcript expectations, and an
independent pixel check covers the guest-drawn header and cursor.

The compiled graph and Wasmtime compared full pages byte for byte at
every update and drawing boundary. Chromium checked both full-page
hashes, all framebuffer bytes and the terminal transcripts. Packed
fallback was blocked. The new histories comprise 46 ticks, or 92 phase
boundaries, alongside the preserved Passage Relay proofs and GIFs.
The installed CPU graph artifact is unchanged.

## Measurements

| Edited history | Offline simulation median / p95 | Offline drawing median / p95 |
| --- | ---: | ---: |
| Rail Crate | 2.33 / 2.44 ms | 11.25 / 11.92 ms |
| Machine Monitor | 2.38 / 13.04 ms | 111.36 / 329.67 ms |

Chromium's 29 keyboard events had **0.40 ms p95**
to Canvas submission and **16.80 ms p95**
to the next animation callback.

VisuaML phase durations exclude Wasmtime work, framebuffer copies,
and hashing. Rail Crate has three warm ticks per history. The Monitor's
drawing includes all affected guest text rows and has 20 warm key
events in the edited log. Chromium key timing dispatches keyboard
events through the focused player and measures through synchronous
Canvas submission. The next animation callback is a display
opportunity. [Offline samples and editor regression](visuaml.json) ·
[Chromium samples and replay results](chromium.json).

Passage Relay's previous 5.49 ms offline p95 applies to Passage Relay's
paced scene. The Monitor's browser timing and offline phase timing
describe different paths.

To capture the GIFs again, start the dev server and run
`node tools/cartridge_browser.mjs /tmp/pocket-cartridges` with Playwright
Core and Chromium, then
`python tools/pack_cartridge_gifs.py /tmp/pocket-cartridges evidence/cartridges`
with Pillow.
