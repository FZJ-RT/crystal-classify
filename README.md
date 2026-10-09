# crystal-classify

Rule-based detection and classification of CaCO₃ and (Ba,Ca)CO₃ crystals in very large optical micrographs.

The script finds every crystal in an image, separates crystals that touch, and sorts each object into one of four classes:

| Class | Meaning |
| --- | --- |
| `vaterite` | Spherulites: near-circular outline with a radially organised internal texture |
| `other` | Rhombohedral calcite, dumbbells and other single crystals |
| `cluster` | Intergrown aggregates that could not be split into single crystals |
| `discarded` | Objects cut by the image border, whose size and shape are unreliable |

It was written for stitched micrographs larger than 1 GB. TIFF files are read in chunks, so the full image is never loaded into memory.

This code accompanies the article *[title]*, [authors], [journal] ([year]), [DOI].

## Installation

Requires Python 3 (3.10 or newer recommended).

```
git clone https://github.com/<your-username>/crystal-classify.git
cd crystal-classify
pip install -r requirements.txt
```

## Usage

```
python crystal_classify.py IMAGE [IMAGE ...] [options]
```

Examples:

```
# one image with a pixel size of 0.185 µm
python crystal_classify.py sample.tif --pixel-size 0.185

# all TIFF files in a folder, with results written to a separate folder
python crystal_classify.py "images/Ca_090/*.tif" --pixel-size 0.185 --outdir results/Ca_090

# quick test on the first 50 objects only
python crystal_classify.py sample.tif --limit 50
```

Put wildcard patterns in quotes. The script expands them itself, which is needed in Windows PowerShell. Each image is processed independently, and if one image fails the others still run.

### Output

For each image, the following files are written to the output folder (default `results`):

| File | Content |
| --- | --- |
| `<name>_crystals.csv` | One row per object: id, class, vaterite score, size in px and µm, shape and texture descriptors, position, and a note explaining the decision |
| `<name>_annotated.png` | Downsampled overview with numbered outlines: green = vaterite, orange = other, magenta = cluster, grey = discarded. The numbers match the `id` column of the CSV |
| `<name>_annotated_fullres.tif` | Only with `--annotate-fullres`: full-resolution BigTIFF with a labelled circle around every object |

### Options

All sizes are in pixels of the full-resolution image. `python crystal_classify.py --help` describes every option in detail.

| Option | Default | Meaning |
| --- | --- | --- |
| `--pixel-size` | none | Micrometres per pixel; without it the µm columns stay empty |
| `--outdir` | `results` | Output folder (created if needed) |
| `--min-diam` | 40 | Objects smaller than this are ignored |
| `--overview-width` | 6000 | Maximum width of the downsampled overview |
| `--annotate-fullres` | off | Also write a full-resolution annotated TIFF |
| `--limit` | 0 (all) | Process only the first N candidate objects, for testing |
| `--max-window-px` | 4000 | Candidates larger than this are labelled cluster without full-resolution analysis |
| `--min-circ-hard` | 0.60 | Minimum circularity for vaterite |
| `--max-ar` | 1.45 | Maximum aspect ratio for vaterite |
| `--min-waist` | 0.80 | Waist ratio below which a crystal counts as a dumbbell |
| `--dumbbell-merge-waist` | 0.80 | A two-part region with a broader neck than this is kept as one crystal |
| `--min-solidity` | 0.90 | Touching crystals below this solidity after splitting are labelled cluster |
| `--score-thr` | 0.50 | Minimum vaterite score |
| `--size-prior-px` | 150 | Reference diameter of the size term in the vaterite score |
| `--cluster-diam-px` | automatic | Objects larger than this are labelled cluster |
| `--cluster-outlier-k` | 3.0 | Controls the automatic cluster size limit |
| `--vaterite-min-diam-px` | automatic | Vaterite candidates smaller than this are labelled other |
| `--vaterite-size-percentile` | 90 | Percentile of object sizes used for the automatic vaterite minimum |

## How it works

1. A downsampled overview of the image is built strip by strip.
2. Candidate objects are found on the overview by thresholding a combination of intensity deviation from the background and local texture.
3. Two size limits are derived from the candidate sizes of each image: an upper limit above which objects are treated as clusters, and a lower limit for vaterite.
4. Each candidate is read again at full resolution and segmented. Touching crystals are separated with a watershed transform, and dumbbells are protected from being split in two.
5. For each crystal, the script calculates its diameter, circularity, aspect ratio, solidity, a waist ratio that detects dumbbells, and a structure-tensor score for how radially its internal texture is organised.
6. Fixed rules and a weighted score assign the class.

A full description with all formulas is given in the Supporting Information of the article.

## Limitations

- The thresholds were set on CaCO₃ and (Ba,Ca)CO₃ crystals in our optical micrographs. Other materials, magnifications or imaging conditions will need different values.
- By default only objects larger than the 90th percentile of an image's object sizes can be classified as vaterite. In samples where vaterite makes up more than about 10 % of the crystals, set `--vaterite-min-diam-px` to a fixed value instead.
- The texture score responds to brightness that varies along the radius, such as concentric growth bands.

## Citation

If you use this code, please cite the article above and the archived software version: [Zenodo DOI]. Citation details are also in `CITATION.cff`, and GitHub shows them under "Cite this repository".

## License

MIT, see `LICENSE`.
