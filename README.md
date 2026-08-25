# Reconstruction of Urban Drainage Networks

Detection and correction of anomalous values, and reconstruction of missing
ones, for urban drainage network (UDN) attributes: manhole **rim elevation**,
manhole **invert elevation**, pipe **diameter** and pipe **material**.

This code accompanies a research article submitted to *TODO*:
*"Reconstruction of Missing and Anomalous Attributes in Urban Drainage
Networks: A Unified Detection-and-Correction Framework"*.

## Overview

The framework combines two complementary reconstruction strategies for the elevation:

- **Method 1 — local spatial interpolation.** Estimates a value at a target
  node from its nearby neighbours (Inverse Distance Weighting, local linear
  regression, or local quadratic regression). Used to detect anomalies (large
  deviation between observed and predicted value), correct them, and predict
  missing values, for both rim and invert elevation. For invert elevation,
  two optional physical constraints (depth, tied to rim elevation; slope,
  tied to the pipe topology) keep predictions hydraulically plausible.
- **Method 2 — topological traversal.** Estimates invert elevation by walking
  the directed pipe graph upstream/downstream to the nearest known values.
  Used where missing values are spatially concentrated rather than scattered,
  so that spatial interpolation has no local neighbour to rely on.

The framework is evaluated on four real-world networks: **Montpellier** and
**Angers** (France), **Kapiti Coast** and **Auckland** (New Zealand).

Diameter and material are also reconstructed for Montpellier only, using a topological score method.

## Repository structure

```
.
├── README.md
├── LICENSE
├── Code/
│   ├── function.py                           # shared reconstruction library (all methods)
│   ├── rimElevation.ipynb                    # rim elevation: correction + prediction, all 4 networks
│   ├── invertElevation.ipynb                 # invert elevation: Method 1 (all networks) + Method 2 (Montpellier)
│   ├── diameter_material_Montpellier.ipynb   # diameter and material: correction + prediction (Montpellier)
│   ├── Montpellier.ipynb                     # Montpellier-specific preprocessing
│   ├── Angers.ipynb                          # Angers-specific preprocessing
│   ├── Auckland.ipynb                        # Auckland-specific preprocessing
│   └── KapitiCoast.ipynb                     # Kapiti Coast-specific preprocessing
└── data/
    ├── Montpellier/
    │   └── SewerRDF_Montpellier.ttl
    ├── Angers/
    │   ├── Wastewater/
    │   │   ├── Angers_Wastewater_Pipes.csv
    │   │   └── Angers_Wastewater_Manholes.csv
    │   └── Stormwater/
    │       ├── Angers_Stormwater_Pipes.csv
    │       └── Angers_Stormwater_Manholes.csv
    ├── Kapiti/
    │   ├── KCDC_Wastewater_Pipes.csv
    │   └── KCDC_Wastewater_Manholes.csv
    └── Auckland/
        ├── Wastewater/
        │   ├── Wastewater_Pipe.csv
        │   └── Wastewater_Manhole.csv
        └── Stormwater/
            ├── Stormwater_Pipe.csv
            └── Stormwater_Manhole.csv
```

Each network is selected inside a notebook via a `choice`/`network` variable
at the top; the reconstruction logic itself lives entirely in
`Code/function.py` and is shared across networks and notebooks
(`from function import *`).

## Data & Licensing

Each dataset under `data/` originates from a different provider and is
released here **unmodified**, under its own original licence. Attribution
is required for all four sources; please keep this section (or the
corresponding sub-notice) attached whenever this data is redistributed.

| Network | Source | Licence |
|---|---|---|
| Montpellier | [Montpellier Méditerranée Métropole open data](https://data.montpellier3m.fr/) (RDF graph adapted by Batoul Haydar) | [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| Angers | [Opendata Angers Loire Métropole](https://data.angers.fr/) | [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| Kapiti Coast | [Kāpiti Coast District Council Open GIS Data](https://data-kcdc.opendata.arcgis.com/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Auckland — Wastewater | Watercare An Auckland Council Organisation | [CC BY-NC-ND 3.0 New Zealand](https://creativecommons.org/licenses/by-nc-nd/3.0/nz/) |
| Auckland — Stormwater | Auckland Council Open Data | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

**Montpellier.** `SewerRDF_nouveau.ttl` is a Derivative Database
built from the [Montpellier Méditerranée Métropole](https://data.montpellier3m.fr/)
sanitation network open data. The RDF adaptation was carried out by
**Batoul Haydar** as part of an earlier project; see:

> Haydar, B.; Chahinian, N.; Pasquier, C. *Reconstructing Sewer Network
> Topology Using Graph Theory.* Water **2026**, 18(2), 222.
> https://doi.org/10.3390/w18020222

As required by ODbL, this file remains licensed under ODbL v1.0; any further
adapted/derivative version published from it must remain under the same
licence, with attribution to both Montpellier Méditerranée Métropole and the
above publication.

**Auckland.** The two `Wastewater_*` files come from Watercare Services
Limited and are included unmodified, as required by the CC BY-NC-ND 3.0 NZ
terms: **no adapted/derivative version of these specific files is
redistributed** in this repository, use is non-commercial, and attribution
to Watercare Services Limited is retained. The two `Stormwater_*` files come
from a different source — Auckland Council Open Data — and are licensed
under CC BY 4.0, the same permissive terms as Kapiti Coast: reuse and
adaptation are permitted with attribution.

**Code licence.** The code in Code/ (all .py and .ipynb files) is independent of the above data licences and is distributed under CeCILL-B v1.0 — see LICENSE at the repository root. Reconstructed attribute tables produced by running this code on the Angers/Montpellier data (ODbL sources) are themselves Derivative Databases and, if published, must remain under ODbL v1.0 with attribution to the original source.

## Requirements

```
pandas==2.3.3
numpy==2.3.5
scipy==1.16.3
matplotlib==3.10.6
geopandas==1.1.3
contextily==1.7.0
rdflib==7.6.0
shapely==2.1.2
rasterio==1.5.0
networkx==3.5
scikit-learn==1.7.2
joblib==1.5.2
ipykernel==6.31.0
jupyter_client==8.6.3
```

## Usage

1. Place the pre-processed pipe/node tables for a network under
   `data/<City>/` (see the corresponding preprocessing notebook for the
   expected format).
2. Open `Code/rimElevation.ipynb`, set the `choice` variable to the target
   network, and run all cells to detect, correct, and predict rim elevation.
3. Open `Code/invertElevation.ipynb` (requires the rim-elevation output from
   step 2) to reconstruct invert elevation.
4. Open `Code/diameter_material_Montpellier.ipynb` (only for Montpellier) to
   reconstruct diameter and material.

## Citation

If you use this code, please cite:

> Vieville S., Pasquier C., Da Costa Pereira C., Tettamanzi A., Chahinian N.
> *Reconstruction of Missing and Anomalous Attributes in Urban Drainage
> Networks: A Unified Detection-and-Correction Framework.* Submitted to
> *TODO*, 2026.

If you use the Montpellier RDF-derived data (`data/Montpellier/`), please
also cite:

> Haydar, B.; Chahinian, N.; Pasquier, C. *Reconstructing Sewer Network
> Topology Using Graph Theory.* Water **2026**, 18(2), 222.
> https://doi.org/10.3390/w18020222

## License

The code in this repository (Code/) is licensed under CeCILL-B, a French free-software license drafted by CEA, CNRS and INRIA, fully compatible in spirit with BSD/MIT-style licenses. See the "Data & Licensing" section above for the data licences, which are separate and unmodified from their original sources.
