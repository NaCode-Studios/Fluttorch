# Fluttorch brand assets

The mark is The Seam: two tongues of flame sharing one base, cut by a single closed seam. The warm
tongue is the training side, the cool one is the device, and the seam between them is the boundary
Fluttorch verifies. It is one shape rather than two overlapping ones, and the seam closes, because a
model that agrees with its reference is one thing and not two.

| File | Use |
| --- | --- |
| `fluttorch-lockup-light.svg` | the logo on a light surface |
| `fluttorch-lockup-dark.svg` | the logo on a dark surface |
| `fluttorch-lockup-mono-light.svg` / `-mono-dark.svg` | one flat ink, inherits `currentColor` |
| `fluttorch-symbol.svg` / `fluttorch-symbol-mono.svg` | the mark on its own, never as the logo |
| `fluttorch-favicon.svg` / `-light.svg` / `fluttorch-favicon-512.png` | favicon, avatar, app icon |
| `fluttorch-social-preview.png` | the GitHub social preview (2560×1280) |
| `fluttorch-tokens.css` | colour, type, space, radius and elevation tokens |
| [`../fluttorch-hero.png`](../fluttorch-hero.png) | the README banner (2560×680) |

Every SVG here is self-contained. The wordmark is outlined to paths, so it renders identically
whether or not Space Grotesk is installed, and nothing references an external font, image or
gradient.

## Construction

The mark lives in a `21 4 60 86` box, two cubic paths and no strokes. In the lockup the mark is 80
units tall against a 96-unit band, the wordmark is Space Grotesk 600 at cap 79 with tracking
`-0.04em`, and the gap between them is half the cap height. That is also the clear space: half the
cap height on every side, measured from the mark rather than from the canvas. The canvas is tight to
the content at 420×96, so there is no phantom padding to fight in a README.

## Minimum size

24px for the two-tone mark. Below that the seam stops reading and only `fluttorch-symbol-mono.svg`
should be used. The lockup has a floor of 96px wide.

## Type

Space Grotesk 600 names things: the wordmark and headings. IBM Plex Sans carries prose. JetBrains
Mono carries every value a machine produced, including tensor names, shapes, hashes, deltas and
backends. If a number came out of a model, it is set in mono.

## Colour

Ember `#EE4C2C` is the training side and the primary action. Signal `#0553B1` is the device side,
links and information, and becomes Beacon `#13B9FD` on dark surfaces. Everything else is the warm
Graphite ramp, from Paper `#FBF9F7` to `#100E0C`. Status has its own family and never borrows from
the brand ramps: Pass `#0E8F5E`, Drift `#B4690E`, Fail `#A81E39`, wine rather than ember, so that a
red build never reads as branding.

## Don't

Use the mark alone as the logo · separate the two tongues or open the seam · rotate, skew or stretch
either part · recolour the mark outside Ember and Signal · put Ember on Signal or Signal on Ember ·
set the wordmark in title case or capitals · substitute the typeface or change the tracking · add a
shadow, an outline or a gradient · use Fail red as an accent.
