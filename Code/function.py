"""
Shared reconstruction library for urban drainage network (UDN) attributes.

This module gathers every function used by the reconstruction notebooks
(rimElevation.ipynb, invertElevation.ipynb). It is
imported wholesale (`from function import *`) so that the notebooks contain
only orchestration code -- network selection, hyperparameter grids, plotting
-- while all the reconstruction logic lives here.

The module implements two complementary families of methods:

* **Spatial interpolation (Method 1)** -- estimates an elevation at a target
  node from its spatial neighbours within a search radius, using one of three
  local estimators of increasing polynomial degree (IDW, local linear
  regression, local quadratic regression). Used to detect anomalous recorded
  values (large deviation between observed and predicted), correct them, and
  predict genuinely missing ones. For invert elevation, two optional physical
  constraints restrict implausible predictions: a *depth* constraint tied to
  the (already corrected) rim elevation, and a *slope* constraint tied to the
  upstream/downstream topological neighbours.

* **Topological traversal (Method 2)** -- estimates an invert elevation by
  walking the directed pipe graph upstream and downstream until nodes with a
  known value are reached, then interpolating between them. Used where
  spatial interpolation has no known neighbour to work from, typically when
  missing values are spatially concentrated rather than scattered.

Evaluation relies on controlled perturbation: anomalies are injected into
known values (`injectAnomalie`) or known values are masked (`hide_invert`),
the pipeline is run, and the reconstruction is scored against the held-out
truth. Where an external elevation reference is available (rim elevation),
`evaluate_real_case` compares the reconstruction against it directly.

Conventions used throughout:
    - node tables carry `node_id`, `x`, `y` (projected metric CRS),
      `rimElevation` and `invertElevation`;
    - `aberrants` / `detected` / `NaN` / `NaN_corrected` are per-node integer
      flags added by the pipeline, not source data;
    - the graph `G` is a directed NetworkX graph whose nodes carry
      `rimElevation` / `invertElevation` attributes and whose edges carry a
      `length` attribute.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import random
from scipy.spatial import cKDTree
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

def discardHeightImpossible(df, feature, minimum, maximum):
    """Replace physically impossible values by NaN.

    Values outside the admissible range are set to NaN rather than to an
    arbitrary substitute, so that they become targets for the reconstruction
    stage instead of silently biasing it. Bounds are network-specific, since a
    plausible elevation in one region may be impossible in another.

    Returns the cleaned `feature` column (the input DataFrame is modified in
    place).
    """
    mask = (df[feature] <= minimum) | (df[feature] >= maximum)
    df.loc[mask, feature] = np.nan
    return df[feature]

def boxplot_mat_dia(df_pipe):
    """Plot the diameter distribution per grouped material class.

    Used to check that the material grouping yields classes with visibly
    distinct diameter distributions -- the property later exploited by the
    diameter/material compatibility scoring.
    """
    plt.figure()

    df_pipe.boxplot(column="diameter", by="material")
    
    plt.title("Diameter by material")
    plt.suptitle("")
    plt.xlabel("Material")
    plt.ylabel("Diameter")
    plt.xticks(rotation=45)
    plt.show()

def precompute_neighbors(df, radius, min_dist=5):
    """Build the spatial neighbourhood of every node once, with a KD-tree.

    Returns two lists, aligned with the rows of `df`: the neighbour indices
    and the corresponding Euclidean distances, for the neighbours lying within
    `radius` metres and further than `min_dist` metres away.

    Excluding near-coincident points (`min_dist`) limits the influence of
    residual duplicates and of distinct elements sharing a location whose
    surveyed elevations may differ; it also prevents such points from
    dominating the distance-weighted estimators, where a very small distance
    would translate into an excessive weight.

    Computing the neighbourhoods once and reusing them across the successive
    detection/correction passes is a major speed-up, since the node positions
    never change during a run.
    """
    coords = df[["x", "y"]].to_numpy(dtype=float)

    tree = cKDTree(coords)

    # for each point, return the indices of the neighbours within the radius
    raw_neighbors = tree.query_ball_point(coords, r=radius)

    neighbors_idx = []
    neighbors_dist = []

    for i, idxs in enumerate(raw_neighbors):
        idxs = np.array(idxs, dtype=int)

        # drop the point itself
        idxs = idxs[idxs != i]

        if idxs.size == 0:
            neighbors_idx.append(np.array([], dtype=int))
            neighbors_dist.append(np.array([], dtype=float))
            continue

        # Euclidean distance in metres
        d = np.linalg.norm(coords[idxs] - coords[i], axis=1)

        # keep only the neighbours that are not too close
        keep = d > min_dist

        neighbors_idx.append(idxs[keep])
        neighbors_dist.append(d[keep])

    return neighbors_idx, neighbors_dist

def calcul_z(method, weight, dists, x0, y0, x_nei, y_nei, z_nei):
    """Estimate the elevation at a target point from its neighbours.

    Implements the three local interpolation methods, which can be read as
    instances of the Moving Least Squares framework of degree 0, 1 and 2:

    * ``IDW``  -- inverse distance weighting: a weighted average of the
      neighbour elevations (degree-0, locally constant fit);
    * ``LR``   -- local linear regression: fits a plane to the neighbourhood;
    * ``QR``   -- local quadratic regression: adds curvature terms, better
      representing non-linear relief.

    For LR and QR the prediction is the intercept of the fitted surface, i.e.
    its value at the target point itself. `weight` selects the weighted
    least-squares scheme (None = uniform).
    """
    if method == "IDW":
        # ── IDW ──────────────────────────────────────────────────────────────
        weight = 1.0 / dists
        return(np.sum(weight * z_nei) / np.sum(weight))

    dx = x_nei - x0
    dy = y_nei - y0

    if method == "LR":
        # ── Local linear regression ──────────────────────────────────────────────────────────      
        X_train = np.column_stack((np.ones(len(dx)), dx, dy))  

    elif method == "QR":
        # ── Local quadratic regression ─────────────────────────────────────────────────────── 
        X_train = np.column_stack((np.ones(len(dx)), dx, dy, dx**2, dy**2, dx * dy))      
        
    # weighted least squares:
    # min || sqrt(W) (Xb - y) ||²
    if weight is not None:
        sw = np.sqrt(weight)
        Xw = X_train * sw[:, None]
        yw = z_nei * sw
    else:
        Xw = X_train
        yw = z_nei

    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    # at the target point dx=0, dy=0, so the prediction is the intercept
    return coef[0]

def clean_df_elev(df_elev, RADIUS, SEUIL, node_min, method, weight_method = "inverse", feature = "rimElevation", verbose = False):
    """Build a reference dataset by discarding every node flagged as anomalous.

    Runs a single detection pass and *removes* -- rather than corrects -- every
    node whose observed value deviates from its spatial prediction by more than
    `SEUIL`, as well as every node with fewer than `node_min` neighbours. The
    result is used as the reference dataset for the injected-anomaly protocol:
    anomalies are injected into data made as reliable as possible beforehand,
    so that the injected errors are the only ones the model should find.
    """
    df_elev = df_elev.copy()
    df_elev = df_elev.dropna(subset=["x", "y", feature])
    df_elev["aberrants"] = 0
    df_elev = df_elev.reset_index(drop=True)

    x_arr = df_elev["x"].to_numpy(dtype=float)    # convert to numpy: the per-node loop below is far too slow on pandas objects
    y_arr = df_elev["y"].to_numpy(dtype=float)
    z_arr = df_elev[feature].to_numpy(dtype=float)
    ab_arr = df_elev["aberrants"].to_numpy(dtype=np.int8)

    neighbors_idx, neighbors_dist = precompute_neighbors(df_elev, RADIUS)    
    
    for i in range(len(df_elev)):
        idxs = neighbors_idx[i]
        dists = neighbors_dist[i]
        n = len(idxs)
        if n < node_min:
            ab_arr[i] = 1
            continue
            
        if weight_method == "inverse" or method == "IDW":
            weight = 1.0 / dists
        elif weight_method == "inverseSquared":
            weight = 1.0 / (dists**2)
        else:
            weight = None

        z_pred = calcul_z(method, weight, dists, x_arr[i], y_arr[i], x_arr[idxs], y_arr[idxs], z_arr[idxs])
        
        if abs(z_pred - z_arr[i]) > SEUIL:
            ab_arr[i] = 1

    df_elev["aberrants"] = ab_arr
    mask = df_elev["aberrants"] == 0
    df_elev_clean = df_elev[mask]
    df_elev_clean = df_elev_clean.reset_index(drop=True)
    
    if verbose:
        print("Anomalous nodes identified and removed: ", len(df_elev) - len(df_elev_clean))

    return(df_elev_clean)

def injectAnomalie(df_clean, err_min = 0, err_max = 120, pourcent = 0.1, feature = "rimElevation"):
    """Inject synthetic anomalies into a fraction of the nodes.

    Replaces the recorded value of `pourcent` of the nodes with a random value
    drawn from `[err_min, err_max]`, an interval calibrated to the elevation
    range of the network. A draw is rejected and repeated when it lands within
    3 m of the original value, so that every selected node truly receives a
    meaningful anomaly rather than a near-identical replacement.

    Returns the perturbed table (with a `modif` flag marking the injected
    nodes) and a table of the original values, kept as ground truth for the
    evaluation.

    The generator is seeded (42) so that every experiment is reproducible.
    """
    X = df_clean.copy()
    modified_idx = []
    X["modif"] = 0
    nb_error = 0
    rng = np.random.default_rng(42)

    while(nb_error < pourcent * len(X)):
        id = rng.integers(0, len(X))
        if X.loc[id, "modif"] == 0: # only if the node has not been modified yet
            error = rng.integers(err_min, err_max)
            if abs(X.loc[id, feature] - error) > 3: # avoid creating anomalies too small to be meaningful
                modified_idx.append(id)
                X.loc[id, "modif"] = 1
                X.loc[id, feature] = error
                nb_error += 1
    
    y = df_clean.loc[modified_idx, ["node_id", feature]].copy()

    return X, y

def detect_and_correct(X, RADIUS, SEUIL, node_min, method,G = None, weight_method = "inverse", pred_rows = None, feature ="rimElevation",
                        restriRim = False, restriInvert = False, depth_min = 0, depth_max = 0, depth_median = 0, alpha = 0,
                        slope_min = 0, slope_max = 0, slope_median = 0, neighbors_idx=None, neighbors_dist=None, verbose = False):
    """Detect and correct anomalous values in two successive passes.

    **First pass (detection).** For every node holding a value, a prediction is
    computed from its neighbourhood and the node is flagged as anomalous when
    the absolute deviation exceeds `SEUIL`.

    **Second pass (correction).** Each flagged node is re-predicted using only
    the neighbours that are still valid -- neither flagged in the first pass
    nor missing -- so that anomalies do not correct one another. When too few
    valid neighbours remain, no correction is proposed and the node is left
    untouched, to avoid unstable reconstructions.

    For `feature="invertElevation"`, two optional physical constraints filter
    and regularise each correction (see `restrictionRimInvert` and
    `restrictionInitTerm`): a prediction scoring 0 is rejected outright, and a
    prediction scoring between 0 and 1 is pulled towards the physically
    expected value with an amplitude controlled by `alpha`. When several
    constraints apply at once and pull in different directions,
    `bestCompromise` searches for the value maximising their product.

    The reason a correction was refused is recorded per node
    (`notCorrectedRim`, `notCorrectedInvert`, `notCorrectedNeighbors`), which
    lets the evaluation distinguish a detection the model deliberately declined
    to act on from one it never made.

    This function is normally called repeatedly over a decreasing sequence of
    thresholds (see `modelWithInjectedErrors`): applying a small threshold from
    the outset produces many false positives, because a strong error perturbs
    the regression of its neighbours and leads to wrongly flagging sound
    points. Correcting the large deviations first mitigates this.

    Returns the updated table and the running list of correction records.
    """
    if pred_rows is None:
        pred_rows = []
        
    X = X.copy()
    X["aberrants"] = 0
    if "modif" not in X.columns:
        X["modif"] = 0
    if "detected" not in X.columns:
        X["detected"] = 0
    X["NaN"] = 0
    if "notCorrectedRim" not in X.columns:
        X["notCorrectedRim"] = 0
    if "notCorrectedInvert" not in X.columns:
        X["notCorrectedInvert"] = 0
    if "notCorrectedNeighbors" not in X.columns:
        X["notCorrectedNeighbors"] = 0

    if neighbors_idx is None or neighbors_dist is None:
        neighbors_idx, neighbors_dist = precompute_neighbors(X, radius=RADIUS)

    mask = X[feature].isna()
    X.loc[mask, "NaN"] = 1

    x_arr = X["x"].to_numpy(dtype=float)    # convert to numpy: the per-node loop below is far too slow on pandas objects
    y_arr = X["y"].to_numpy(dtype=float)
    z_arr = X[feature].to_numpy(dtype=float)
    ab_arr = X["aberrants"].to_numpy(dtype=np.int8)
    modif_arr = X["modif"].to_numpy(dtype=np.int8)
    detect_arr = X["detected"].to_numpy(dtype=np.int8)
    nan_arr = X["NaN"].to_numpy(dtype=np.int8)
    node_ids = X["node_id"].to_numpy()
    notCorrRim_arr = X["notCorrectedRim"].to_numpy(dtype=np.int8)
    notCorrInvert_arr = X["notCorrectedInvert"].to_numpy(dtype=np.int8)
    notCorrNeighbors_arr = X["notCorrectedNeighbors"].to_numpy(dtype=np.int8)
    rim_arr = X["rimElevation"].to_numpy(dtype=float)
    

    
    for i in range(len(X)):
        if ab_arr[i] != 1 and nan_arr[i] != 1:
            idxs = neighbors_idx[i]
            dists = neighbors_dist[i]
            
            # ── Exclude the neighbours whose value is missing ─────
            valid = nan_arr[idxs] != 1
            idxs = idxs[valid]
            dists = dists[valid]
            
            n = len(idxs)
            if n < node_min:
                continue
    
            if weight_method == "inverse" or method == "IDW":
                weight = 1.0 / dists
            elif weight_method == "inverseSquared":
                weight = 1.0 / (dists**2)
            else:
                weight = None
        
            z_pred = calcul_z(method, weight, dists, x_arr[i], y_arr[i], x_arr[idxs], y_arr[idxs], z_arr[idxs])

            if abs(z_pred - z_arr[i]) > SEUIL:
                ab_arr[i] = 1
                detect_arr[i] = 1
    if verbose:
        print("Anomalous nodes identified: ", ab_arr.sum())

    corr_idx = []
    corr_val = []

    aberrant_set = set()           # anomalous node ids, passed to restrictionInitTerm so that
                                   # the topological traversal skips them
    for k in range(len(ab_arr)):
        if ab_arr[k] == 1:
            aberrant_set.add(node_ids[k])

    for i in range(len(X)):
        if ab_arr[i] == 1:
            idxs = neighbors_idx[i]
            dists = neighbors_dist[i]
        
            # ── Exclude the neighbours that are themselves anomalous or missing ───
            valid = (ab_arr[idxs] != 1) & (nan_arr[idxs] != 1)
            idxs = idxs[valid]
            dists = dists[valid]
        
            n_valid = len(idxs)
            if n_valid < node_min:
                notCorrNeighbors_arr[i] = 1
                # not enough valid neighbours -> leave the node uncorrected
                continue

            if weight_method == "inverse" or method == "IDW":
                weight = 1.0 / dists
            elif weight_method == "inverseSquared":
                weight = 1.0 / (dists**2)
            else:
                weight = None
    
            z_pred = calcul_z(method, weight, dists, x_arr[i], y_arr[i], x_arr[idxs], y_arr[idxs], z_arr[idxs])

            if feature == "invertElevation":
                
                if restriInvert:
                    (score_upstream, score_downstream, upstream_invert_mean, upstream_length_mean, downstream_invert_mean, downstream_length_mean) = restrictionInitTerm(G, node_ids[i], z_pred,
                                                                                                                                                        aberrant_set, slope_median, slope_min, slope_max)
                if restriRim:
                    rimScore = restrictionRimInvert(rim_arr[i], z_pred, depth_min, depth_max, depth_median)
                
                if restriRim and restriInvert:
                    if rimScore is None and score_upstream is None and score_downstream is None:
                        pass            #If all scores are None, we cannot apply any restriction, we keep the predicted value as is

                    elif rimScore is not None and score_upstream is None and score_downstream is None:
                        if rimScore == 0: # depth outside the admissible range -> no correction
                            notCorrRim_arr[i] = 1
                            continue
                        target_invert = rim_arr[i] - depth_median
                        z_pred = z_pred + alpha * (1 - rimScore) * (target_invert - z_pred)

                    elif rimScore is None and score_upstream is not None and score_downstream is None:
                        if score_upstream == 0:
                            notCorrInvert_arr[i] = 1
                            continue
                        target_invert = upstream_invert_mean - slope_median * upstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_upstream) * (target_invert - z_pred)
                    
                    elif rimScore is None and score_upstream is None and score_downstream is not None:
                        if score_downstream == 0:
                            notCorrInvert_arr[i] = 1
                            continue
                        target_invert = downstream_invert_mean + slope_median * downstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_downstream) * (target_invert - z_pred)
                    
                    else:  
                        z_pred = bestCompromise(G, rim_arr[i], node_ids[i], depth_min, depth_max, depth_median, slope_median, slope_min, slope_max,
                                                            rimScore, score_upstream, score_downstream, z_pred,
                                                            aberrant_set)
                    if z_pred is None:
                        notCorrRim_arr[i] = 1
                        notCorrInvert_arr[i] = 1
                        continue


                if not restriRim and restriInvert:
                    if score_upstream is not None and score_downstream is not None: 
                        if score_upstream == 0 or score_downstream == 0:
                            notCorrInvert_arr[i] = 1
                            continue
                        else:
                            z_pred = bestCompromise(G, rim_arr[i], node_ids[i], depth_min, depth_max, depth_median, slope_median, slope_min, slope_max,
                                                            None, score_upstream, score_downstream, z_pred,
                                                            aberrant_set)
                    if score_upstream is not None and score_downstream is None:
                        if score_upstream == 0:
                            notCorrInvert_arr[i] = 1
                            continue
                        target_invert = upstream_invert_mean - slope_median * upstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_upstream) * (target_invert - z_pred)

                    if score_upstream is None and score_downstream is not None:
                        if score_downstream == 0:
                            notCorrInvert_arr[i] = 1
                            continue
                        target_invert = downstream_invert_mean + slope_median * downstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_downstream) * (target_invert - z_pred)
                    
                    if z_pred is None:
                        notCorrInvert_arr[i] = 1
                        continue
            

                if restriRim and not restriInvert:
                    if rimScore is not None: # None when the node has no rim elevation to constrain against
                        if rimScore == 0:    # depth outside the admissible range -> no correction
                            notCorrRim_arr[i] = 1
                            continue
                        target_invert = rim_arr[i] - depth_median
                        z_pred = z_pred + alpha * (1 - rimScore) * (target_invert - z_pred)


            corr_idx.append(i)
            corr_val.append(z_pred)
            notCorrRim_arr[i] = 0
            notCorrInvert_arr[i] = 0
            notCorrNeighbors_arr[i] = 0


            pred_rows.append({
                "node_id": node_ids[i],
                "original": z_arr[i],
                "corrected": round(z_pred, 3),
                "n_voisins": len(idxs),
                "modif": modif_arr[i]
            })

    if verbose:
        print("Anomalous nodes corrected: ", len(corr_idx))

    if len(corr_idx) > 0:
        z_arr[corr_idx] = corr_val
        ab_arr[corr_idx] = 0

    X[feature] = z_arr
    X["aberrants"] = ab_arr
    X["detected"] = detect_arr
    X["notCorrectedRim"] = notCorrRim_arr
    X["notCorrectedInvert"] = notCorrInvert_arr
    X["notCorrectedNeighbors"] = notCorrNeighbors_arr

    return X, pred_rows

def evaluateReconstruction(X, y, pred_rows, RADIUS, SEUIL, node_min, method,df_clean, weight_method = "inverse", err_min = 0, err_max = 120, pourcent = 0.1, feature="rimElevation",
                            restriRim = False, restriInvert = False, depth_min=0, depth_max=0, depth_median=0, alpha=0, slope_min=0, slope_max=0, slope_median=0, verbose = False):
    """Score a run of the injected-anomaly protocol.

    Evaluation covers two distinct aspects:

    * **Detection** -- each injected node is an anomaly the model should
      recover. Precision, recall, F1 and the Matthews correlation coefficient
      (MCC) are computed. The MCC is reported because, unlike F1, it accounts
      for true negatives and therefore collapses towards zero for a model that
      flags nearly everything as anomalous -- a failure mode F1 hides at high
      injection rates.

    * **Reconstruction quality** -- the mean absolute and root-mean-square
      errors, computed separately for true positives (injected nodes correctly
      corrected), false positives (sound nodes corrected in error, scored
      against their pre-injection value) and false negatives (injected nodes
      left uncorrected). Separating them quantifies both the ability to repair
      genuine anomalies and any tendency to degrade sound data.

    Detection metrics are additionally reported under several definitions of
    what counts as a positive, obtained by subtracting the `notCorrected*`
    flags from the raw detections: this distinguishes nodes the model never
    flagged from nodes it flagged but deliberately declined to correct because
    a physical constraint or a lack of valid neighbours vetoed the correction.

    Returns a flat dict combining the hyperparameters and every metric, ready
    to be appended to a grid-search results table.
    """
    initial_lookup = dict(zip(df_clean["node_id"], 
                              df_clean[feature]))

    predictions = (pd.DataFrame(pred_rows).drop_duplicates(subset="node_id", keep="last")) # a node may be corrected in several passes: keep only its last correction
    predictions["delta"] = (predictions["original"] - predictions["corrected"]).abs()


    y_true = X["modif"].astype(int)
    y_pred = X["detected"].astype(int)
    y_notCorrRim = X["notCorrectedRim"].astype(int)
    y_notCorrInvert = X["notCorrectedInvert"].astype(int)
    y_notCorrNeighbors = X["notCorrectedNeighbors"].astype(int)
    y_notCorrectedRestri = (y_notCorrRim + y_notCorrInvert).astype(int)
    y_notCorrectedRestri[y_notCorrectedRestri == 2] = 1
    y_notCorrected = (y_notCorrectedRestri + y_notCorrNeighbors).astype(int)
    y_notCorrected[y_notCorrected == 2] = 1

    n_nan_remaining = (X["NaN"] == 1).sum()
    if "NaN_corrected" in X.columns:
        n_nan_corrected = (X["NaN_corrected"] == 1).sum()
    else:
        n_nan_corrected = 0


    if verbose:
        print(f"Corrected nodes  : {len(predictions)}")
        print(f"Missing nodes remaining : {n_nan_remaining}")
        print(f"Missing nodes reconstructed : {n_nan_corrected}")
        print(f"Percentage of missing nodes reconstructed : {n_nan_corrected / (n_nan_remaining + n_nan_corrected):.2%}")
        print()
        
        print("Threshold = ",SEUIL)
        print("Radius = ",RADIUS)
        print("Min. neighbours required for a correction = ",node_min)
        print("Injected errors range : ",err_min," - ", err_max)
        print("Model : ",method)
                
        
        print("── Distribution of the corrections ──")
        print(f"  Mean delta  : {predictions['delta'].mean():.1f}m")
        print(f"  Median delta: {predictions['delta'].median():.1f}m")
        print(f"  Max delta   : {predictions['delta'].max():.1f}m")
        print(f" Median number of neighbours : {predictions['n_voisins'].median()}")
        print()
        print("── Ten largest corrections ──")
        print(predictions.sort_values("delta", ascending=False)
            .head(10)[["node_id", "original","corrected","delta","n_voisins"]]
            .round(2).to_string())
    

        print()
        print("Classification result only on the detection")
        print("Precision         :", round(precision_score(y_true, y_pred),3))
        print("Recall            :", round(recall_score(y_true, y_pred),3))
        print("F1-score          :", round(f1_score(y_true, y_pred),3))
        print("Matthews Correlation Coefficient :", round(matthews_corrcoef(y_true, y_pred),3))
        print()
        print("Confusion matrix :")
        print(confusion_matrix(y_true, y_pred))

        if feature == "invertElevation":
            print()
            print("Classification result with the restriction on RimElevation") 
            print("Precision         :", round(precision_score(y_true, y_pred - y_notCorrRim),3))
            print("Recall            :", round(recall_score(y_true, y_pred - y_notCorrRim),3))
            print("F1-score          :", round(f1_score(y_true, y_pred - y_notCorrRim),3))
            print("Matthews Correlation Coefficient :", round(matthews_corrcoef(y_true, y_pred - y_notCorrRim),3))
            print()
            print("Confusion matrix :")
            print(confusion_matrix(y_true, y_pred - y_notCorrRim))

            """
            print()
            print("Classification result with the restriction on InvertElevation")
            print("Precision         :", round(precision_score(y_true, y_pred - y_notCorrInvert),3))
            print("Recall            :", round(recall_score(y_true, y_pred - y_notCorrInvert),3))
            print("F1-score          :", round(f1_score(y_true, y_pred - y_notCorrInvert),3))
            print("Matthews Correlation Coefficient :", round(matthews_corrcoef(y_true, y_pred - y_notCorrInvert),3))
            print()
            print("Confusion matrix :")
            print(confusion_matrix(y_true, y_pred - y_notCorrInvert))
            """
        
        print()
        print("Classification result only on the detection with the node that could not be corrected because of the neighbors")
        print("Precision         :", round(precision_score(y_true, y_pred - y_notCorrNeighbors),3))
        print("Recall            :", round(recall_score(y_true, y_pred - y_notCorrNeighbors),3))
        print("F1-score          :", round(f1_score(y_true, y_pred - y_notCorrNeighbors),3))
        print("Matthews Correlation Coefficient :", round(matthews_corrcoef(y_true, y_pred - y_notCorrNeighbors),3))
        print()
        print("Confusion matrix :")
        print(confusion_matrix(y_true, y_pred - y_notCorrNeighbors))

        if feature == "invertElevation":
            """
            print()
            print("Classification result with both restriction")
            print("Precision         :", round(precision_score(y_true, y_pred - y_notCorrectedRestri),3))
            print("Recall            :", round(recall_score(y_true, y_pred - y_notCorrectedRestri),3))
            print("F1-score          :", round(f1_score(y_true, y_pred - y_notCorrectedRestri),3))
            print("Matthews Correlation Coefficient :", round(matthews_corrcoef(y_true, y_pred - y_notCorrectedRestri),3))
            print()
            print("Confusion matrix :")
            print(confusion_matrix(y_true, y_pred - y_notCorrectedRestri))
            """

            print()
            print("Classification result with restriction and neighbors")
            print("Precision         :", round(precision_score(y_true, y_pred - y_notCorrected),3))
            print("Recall            :", round(recall_score(y_true, y_pred - y_notCorrected),3))
            print("F1-score          :", round(f1_score(y_true, y_pred - y_notCorrected),3))
            print("Matthews Correlation Coefficient :", round(matthews_corrcoef(y_true, y_pred - y_notCorrected),3))
            print()
            print("Confusion matrix :")
            print(confusion_matrix(y_true, y_pred - y_notCorrected))    
    
    MAE_TP = 0
    RMSE_TP = 0
    MAE_FP = 0
    RMSE_FP = 0
    MAE_FN = 0
    RMSE_FN = 0
    
    # --- Prepare the ground-truth tables ---
    y_true_df = y.rename(columns={feature: "true_rim"})
    x_current_df = X[["node_id", feature]].rename(columns={feature: "current_rim"})
    
    # --- Join the true value onto the predictions ---
    eval_pred = predictions.merge(y_true_df, on="node_id", how="left")
    
    # TP = corrected and genuinely injected
    tp_df = eval_pred[eval_pred["true_rim"].notna()].copy()
    
    # FP = corrected although it had not been injected
    fp_df = eval_pred[eval_pred["true_rim"].isna()].copy()
    fp_df["true_value"] = fp_df["node_id"].map(initial_lookup)
    
    # --- True-positive errors ---
    tp_df["abs_err"] = (tp_df["corrected"] - tp_df["true_rim"]).abs()
    tp_df["sq_err"] = (tp_df["corrected"] - tp_df["true_rim"]) ** 2
    
    MAE_TP = tp_df["abs_err"].mean()
    RMSE_TP = np.sqrt(tp_df["sq_err"].mean())

    # --- False-positive errors (scored against the pre-injection value) ---
    fp_df["abs_err"] = (fp_df["corrected"] - fp_df["true_value"]).abs()
    fp_df["sq_err"] = (fp_df["corrected"] - fp_df["true_value"]) ** 2
    
    MAE_FP = fp_df["abs_err"].mean()
    RMSE_FP = np.sqrt(fp_df["sq_err"].mean())
    
    # --- FN = injected but left uncorrected ---
    predicted_ids = set(predictions["node_id"])
    fn_df = y_true_df[~y_true_df["node_id"].isin(predicted_ids)].merge(
        x_current_df,
        on="node_id",
        how="left"
    )

    fn_df["abs_err"] = (fn_df["true_rim"] - fn_df["current_rim"]).abs()
    fn_df["sq_err"] = (fn_df["true_rim"] - fn_df["current_rim"]) ** 2
    
    MAE_FN = fn_df["abs_err"].mean()
    RMSE_FN = np.sqrt(fn_df["sq_err"].mean())
    
    if verbose:
        print("MAE True Positive : ", round(MAE_TP, 3) if pd.notna(MAE_TP) else np.nan)
        print("RMSE True Positive : ", round(RMSE_TP, 3) if pd.notna(RMSE_TP) else np.nan)
        print("MAE False Positive : ", round(MAE_FP, 3) if pd.notna(MAE_FP) else np.nan)
        print("RMSE False Positive : ", round(RMSE_FP, 3) if pd.notna(RMSE_FP) else np.nan)
        print("MAE False Negative : ", round(MAE_FN, 3) if pd.notna(MAE_FN) else np.nan)
        print("RMSE False Negative : ", round(RMSE_FN, 3) if pd.notna(RMSE_FN) else np.nan)

    result = {
    "method": method,
    "radius": RADIUS,
    "seuil": SEUIL,
    "node_min": node_min,
    "weight_method": weight_method,
    "err_min": err_min,
    "err_max": err_max,
    "pct_error": pourcent,
    "feature": feature,
    "restriRim": restriRim,
    "restriInvert": restriInvert,
    "depth_min": depth_min,
    "depth_max": depth_max,
    "depth_median": depth_median,
    "slope_min": slope_min,
    "slope_max": slope_max,
    "slope_median": slope_median,
    "alpha": alpha,
    "n_aberrants_detected": int(X["detected"].sum()),
    "n_corrected": int(len(predictions)),
    "n_nan_remaining": int(n_nan_remaining),
    "n_nan_corrected": int(n_nan_corrected),
    "pct_nan_corrected": float(n_nan_corrected / (n_nan_remaining + n_nan_corrected)),

    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "f1": float(f1_score(y_true, y_pred)),
    "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    
    "precision_rim_restr": float(precision_score(y_true, y_pred - y_notCorrRim)),
    "recall_rim_restr": float(recall_score(y_true, y_pred - y_notCorrRim)),
    "f1_rim_restr": float(f1_score(y_true, y_pred - y_notCorrRim)),
    "matthews_corrcoef_rim_restr": float(matthews_corrcoef(y_true, y_pred - y_notCorrRim)),

    "precision_inv_restr": float(precision_score(y_true, y_pred - y_notCorrInvert)),
    "recall_inv_restr": float(recall_score(y_true, y_pred - y_notCorrInvert)),
    "f1_inv_restr": float(f1_score(y_true, y_pred - y_notCorrInvert)), 
    "matthews_corrcoef_inv_restr": float(matthews_corrcoef(y_true, y_pred - y_notCorrInvert)),

    "precision_neighbors": float(precision_score(y_true, y_pred - y_notCorrNeighbors)),
    "recall_neighbors": float(recall_score(y_true, y_pred - y_notCorrNeighbors)),
    "f1_neighbors": float(f1_score(y_true, y_pred - y_notCorrNeighbors)),
    "matthews_corrcoef_neighbors": float(matthews_corrcoef(y_true, y_pred - y_notCorrNeighbors)),

    "precision_restr": float(precision_score(y_true, y_pred - y_notCorrectedRestri)),
    "recall_restr": float(recall_score(y_true, y_pred - y_notCorrectedRestri)),
    "f1_restr": float(f1_score(y_true, y_pred - y_notCorrectedRestri)),
    "matthews_corrcoef_restr": float(matthews_corrcoef(y_true, y_pred - y_notCorrectedRestri)),

    "precision_restr_neighbors": float(precision_score(y_true, y_pred - y_notCorrected)),
    "recall_restr_neighbors": float(recall_score(y_true, y_pred - y_notCorrected)),
    "f1_restr_neighbors": float(f1_score(y_true, y_pred - y_notCorrected)),
    "matthews_corrcoef_restr_neighbors": float(matthews_corrcoef(y_true, y_pred - y_notCorrected)),

    "MAE_TP": float(MAE_TP),
    "RMSE_TP": float(RMSE_TP),
    "MAE_FP": float(MAE_FP),
    "RMSE_FP": float(RMSE_FP),
    "MAE_FN": float(MAE_FN),
    "RMSE_FN": float(RMSE_FN),
    }

    return result

def predictNaN(X, RADIUS, node_min, method, weight_method = "inverse", pred_rows = None, feature = "rimElevation", G = None,
                restriRim = False, restriInvert = False, depth_min = 0, depth_max = 0, depth_median = 0,
                  alpha = 0, slope_min = 0, slope_max = 0, slope_median = 0, neighbors_idx=None, neighbors_dist=None):
    """Reconstruct the genuinely missing values by spatial interpolation.

    Same interpolation framework and same physical constraints as
    `detect_and_correct`, applied to the nodes flagged `NaN` instead of the
    nodes flagged anomalous. Only neighbours holding a valid value contribute
    -- a value either never flagged, or flagged and since corrected.

    Running this *after* the detection/correction passes is deliberate: the
    interpolation then operates on a cleaned neighbourhood, so residual
    anomalies do not propagate into the reconstructed values.

    The limitation is structural: where every neighbour within the radius is
    itself missing -- typically at the sparse periphery of a network -- no
    local information exists and the value is left missing. The coverage
    actually achieved is therefore reported as an evaluation criterion in its
    own right, alongside the reconstruction error.

    Nodes reconstructed here are marked `NaN_corrected`, which lets the
    evaluation score them separately from the corrected known values.
    """
    if pred_rows is None:
        pred_rows = []
    X = X.copy()

    X["NaN_corrected"] = 0

    if neighbors_idx is None or neighbors_dist is None:
        neighbors_idx, neighbors_dist = precompute_neighbors(X, radius=RADIUS)

    x_arr = X["x"].to_numpy(dtype=float)    # convert to numpy: the per-node loop below is far too slow on pandas objects
    y_arr = X["y"].to_numpy(dtype=float)
    z_arr = X[feature].to_numpy(dtype=float)
    ab_arr = X["aberrants"].to_numpy(dtype=np.int8)
    modif_arr = X["modif"].to_numpy(dtype=np.int8)
    nan_arr = X["NaN"].to_numpy(dtype=np.int8)
    node_ids = X["node_id"].to_numpy()
    nanCorr_arr = X["NaN_corrected"].to_numpy(dtype=np.int8)
    rim_arr = X["rimElevation"].to_numpy(dtype=float)

    corr_idx = []
    corr_val = []
    for i in range(len(X)):
        if nan_arr[i] == 1:
            idxs = neighbors_idx[i]
            dists = neighbors_dist[i]
        
            # ── Exclude the neighbours that are themselves anomalous or missing ───
            valid = (ab_arr[idxs] != 1) & (nan_arr[idxs] != 1)
            idxs = idxs[valid]
            dists = dists[valid]
        
            n_valid = len(idxs)
            if n_valid < node_min:
                # not enough valid neighbours -> leave the node uncorrected
                continue

            if weight_method == "inverse" or method == "IDW":
                weight = 1.0 / dists
            elif weight_method == "inverseSquared":
                weight = 1.0 / (dists**2)
            else:
                weight = None
    
            z_pred = calcul_z(method, weight, dists, x_arr[i], y_arr[i], x_arr[idxs], y_arr[idxs], z_arr[idxs])

            if feature == "invertElevation":
                
                if restriInvert:
                    (score_upstream, score_downstream, upstream_invert_mean, upstream_length_mean, downstream_invert_mean, downstream_length_mean) = restrictionInitTerm(G, node_ids[i], z_pred,
                                                                                                                                                        set(), slope_median, slope_min, slope_max)
                if restriRim:
                    rimScore = restrictionRimInvert(rim_arr[i], z_pred, depth_min, depth_max, depth_median)
                
                if restriRim and restriInvert:
                    if rimScore is None and score_upstream is None and score_downstream is None:
                        pass            #If all scores are None, we cannot apply any restriction, we keep the predicted value as is
                        
                    elif rimScore is not None and score_upstream is None and score_downstream is None:
                        if rimScore == 0: # depth outside the admissible range -> no correction
                            continue
                        target_invert = rim_arr[i] - depth_median
                        z_pred = z_pred + alpha * (1 - rimScore) * (target_invert - z_pred)

                    elif rimScore is None and score_upstream is not None and score_downstream is None:
                        if score_upstream == 0:
                            continue
                        target_invert = upstream_invert_mean - slope_median * upstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_upstream) * (target_invert - z_pred)
                    
                    elif rimScore is None and score_upstream is None and score_downstream is not None:
                        if score_downstream == 0:
                            continue
                        target_invert = downstream_invert_mean + slope_median * downstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_downstream) * (target_invert - z_pred)
                    
                    else:  
                        z_pred = bestCompromise(G, rim_arr[i], node_ids[i], depth_min, depth_max, depth_median, slope_median, slope_min, slope_max,
                                                            rimScore, score_upstream, score_downstream, z_pred,
                                                            set())
                    if z_pred is None:
                        continue


                if not restriRim and restriInvert:
                    if score_upstream is not None and score_downstream is not None: 
                        if score_upstream == 0 or score_downstream == 0:
                            continue
                        else:
                            z_pred = bestCompromise(G, rim_arr[i], node_ids[i], depth_min, depth_max, depth_median, slope_median, slope_min, slope_max,
                                                            None, score_upstream, score_downstream, z_pred,
                                                            set())
                    if score_upstream is not None and score_downstream is None:
                        if score_upstream == 0:
                            continue
                        target_invert = upstream_invert_mean - slope_median * upstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_upstream) * (target_invert - z_pred)

                    if score_upstream is None and score_downstream is not None:
                        if score_downstream == 0:
                            continue
                        target_invert = downstream_invert_mean + slope_median * downstream_length_mean
                        z_pred = z_pred + alpha * (1 - score_downstream) * (target_invert - z_pred)

                    if z_pred is None:
                        continue
            

                if restriRim and not restriInvert:
                    if rimScore is not None: # None when the node has no rim elevation to constrain against
                        if rimScore == 0:    # depth outside the admissible range -> no correction
                            continue
                        target_invert = rim_arr[i] - depth_median
                        z_pred = z_pred + alpha * (1 - rimScore) * (target_invert - z_pred)

            corr_idx.append(i)
            corr_val.append(z_pred)

            pred_rows.append({
                "node_id": node_ids[i],
                "original": z_arr[i],
                "corrected": round(z_pred, 3),
                "n_voisins": len(idxs),
                "modif": modif_arr[i]
            })

    if len(corr_idx) > 0:
        z_arr[corr_idx] = corr_val
        nan_arr[corr_idx] = 0
        nanCorr_arr[corr_idx] = 1

    X[feature] = z_arr
    X["NaN"] = nan_arr
    X["NaN_corrected"] = nanCorr_arr

    return X, pred_rows

def evaluate_real_case(X, z_ref, radius, node_min, method, weight_method, final_seuil, feature="rimElevation", epsg="EPSG:2154", verbose=True):
    """Compare a reconstruction against an external elevation reference.

    Used for rim elevation, the only attribute for which an independent ground
    truth exists (a national LiDAR-derived elevation model). Correcting
    recorded values and predicting missing ones are two distinct tasks, so
    error statistics are reported separately for each group as well as
    globally:

    * `*_known` -- nodes whose value was already recorded and possibly
      corrected; the before/after comparison measures whether the model repairs
      genuine anomalies without degrading sound data;
    * `*_nan`   -- nodes whose value was missing and has been reconstructed;
    * `*_global`-- both groups together.

    Bias (signed mean error) is reported alongside MAE and RMSE to reveal any
    systematic offset with respect to the reference, which a purely absolute
    metric would hide.

    Returns the annotated GeoDataFrame and a flat summary dict.
    """
    if "geometry" not in X.columns:
        X["geometry"] = gpd.points_from_xy(X["x"], X["y"], crs=epsg)

    gdf_after = gpd.GeoDataFrame(X.copy(), geometry="geometry")
    gdf_after = gdf_after.set_crs(epsg)

    gdf_after["z_ref"] = z_ref

    # count the still-missing nodes before dropping them from the comparison
    n_nan_remaining = (gdf_after["NaN"] == 1).sum()
    
    gdf_after = gdf_after.dropna(subset=[feature, "z_ref"]).copy()
    gdf_after["err"] = gdf_after[feature] - gdf_after["z_ref"]
    gdf_after["abs_err"] = gdf_after["err"].abs()

    n_total = len(gdf_after)
    mae_global = gdf_after["abs_err"].mean()
    rmse_global = np.sqrt((gdf_after["err"] ** 2).mean())
    bias_global = gdf_after["err"].mean()
    
    mask = gdf_after["NaN_corrected"] == 0
    predKnown = gdf_after.loc[mask]
    
    mae_known = predKnown["abs_err"].mean()
    rmse_known = np.sqrt((predKnown["err"] ** 2).mean())
    bias_known = predKnown["err"].mean()
    n_known = len(predKnown)
    
    mask = gdf_after["NaN_corrected"] == 1
    predNaN = gdf_after.loc[mask]

    n_nan_corrected = len(predNaN)
    mae_nan = predNaN["abs_err"].mean()
    rmse_nan = np.sqrt((predNaN["err"] ** 2).mean())
    bias_nan = predNaN["err"].mean()

    summary = {
        "method": method,
        "radius": radius,
        "node_min": node_min,
        "weight_method": weight_method,
        "final_seuil": final_seuil,

        "n_nan_remaining": n_nan_remaining,

        "n_nan_corrected": n_nan_corrected,
        "mae_nan": mae_nan,
        "rmse_nan": rmse_nan,
        "bias_nan": bias_nan,

        "n_known": n_known,
        "mae_known": mae_known,
        "rmse_known": rmse_known,
        "bias_known": bias_known,

        "n_total": n_total,
        "mae_global": mae_global,
        "rmse_global": rmse_global,
        "bias_global": bias_global,
    }

    if verbose:
        print(method, radius, node_min)

    return gdf_after, summary


def restrictionRimInvert(rim, invert, depth_min = 0, depth_max=0, depth_median = 0):
    """Score a candidate invert elevation against the plausible manhole depth.

    The depth implied by the candidate (rim minus invert) is scored on a
    triangular function peaking at the median network depth and falling to zero
    at the interval bounds. A score of 0 rejects the candidate outright; a
    score between 0 and 1 keeps it but marks it as marginal, so that the caller
    can pull it towards the median-depth target.

    `depth_min` may be slightly negative in some configurations. Although a
    negative depth is physically impossible, tolerating a small one prevents
    the algorithm from discarding predictions that only appear inconsistent
    because of residual error in the corrected rim elevation.

    Returns None when the node has no rim elevation, i.e. when the constraint
    simply cannot be evaluated -- which the caller must distinguish from a
    score of 0 (constraint evaluated and violated).
    """
    if pd.isna(rim):
        return None  # no rim elevation -> the constraint cannot be evaluated
    
    depth = rim - invert
    if depth < depth_min or depth > depth_max:
        score = 0
    else:
        score = parabolic_score(depth, depth_median, depth_min, depth_max)
    return score

def slope_scores_from_means(invert, upstream_invert_mean, upstream_length_mean,
                              downstream_invert_mean, downstream_length_mean,
                              slope_median, slope_min, slope_max):
    """Compute upstream/downstream slope scores for a candidate invert value,
    from already-computed neighbour means -- no graph traversal here."""
    if upstream_invert_mean is not None and upstream_length_mean is not None:
        upstream_slope = (upstream_invert_mean - invert) / upstream_length_mean
        score_upstream = parabolic_score(upstream_slope, slope_median, slope_min, slope_max)
    else:
        score_upstream = None

    if downstream_invert_mean is not None and downstream_length_mean is not None:
        downstream_slope = (invert - downstream_invert_mean) / downstream_length_mean
        score_downstream = parabolic_score(downstream_slope, slope_median, slope_min, slope_max)
    else:
        score_downstream = None

    return score_upstream, score_downstream


def restrictionInitTerm(G, node_id, invert, aberrant_set, slope_median, slope_min, slope_max):
    """Score a candidate invert elevation against the neighbouring pipe slopes.

    Walks the directed graph upstream and downstream to find the nearest nodes
    holding a known invert elevation, averages their elevations and their
    cumulative distances along the pipes, and scores the resulting upstream and
    downstream slopes on the same triangular function as the depth constraint,
    centred on the median network slope.

    Nodes listed in `aberrant_set` are skipped, so that a value already flagged
    as anomalous cannot serve as a reference for correcting another node.

    Returns the two scores together with the neighbour means, which the caller
    reuses to compute the physically expected target value without traversing
    the graph a second time. Either score is None when no usable neighbour
    exists on that side.
    """
    downstream_invert_list, downstream_length_list = [], []
    upstream_invert_list, upstream_length_list = [], []

    downstreams = nearest_neighbors_downstream(G, node_id)
    if downstreams:
        for downstream in downstreams:
            if downstream[0] in aberrant_set:
                continue
            if pd.notna(downstream[1]) and pd.notna(downstream[2]):
                downstream_invert_list.append(downstream[1])
                downstream_length_list.append(downstream[2])
    if len(downstream_invert_list) > 0:
        downstream_invert_mean = sum(downstream_invert_list) / len(downstream_invert_list)
        downstream_length_mean = sum(downstream_length_list) / len(downstream_length_list)
    else:
        downstream_invert_mean = None
        downstream_length_mean = None

    upstreams = nearest_neighbors_upstream(G, node_id)
    if upstreams:
        for upstream in upstreams:
            if upstream[0] in aberrant_set:
                continue
            if pd.notna(upstream[1]) and pd.notna(upstream[2]):
                upstream_invert_list.append(upstream[1])
                upstream_length_list.append(upstream[2])
    if len(upstream_invert_list) > 0:
        upstream_invert_mean = sum(upstream_invert_list) / len(upstream_invert_list)
        upstream_length_mean = sum(upstream_length_list) / len(upstream_length_list)
    else:
        upstream_invert_mean = None
        upstream_length_mean = None

    score_upstream, score_downstream = slope_scores_from_means(
        invert, upstream_invert_mean, upstream_length_mean,
        downstream_invert_mean, downstream_length_mean,
        slope_median, slope_min, slope_max,
    )

    return (score_upstream, score_downstream, upstream_invert_mean, upstream_length_mean,
            downstream_invert_mean, downstream_length_mean)


def parabolic_score(x, median, min_val, max_val):
    """Triangular plausibility score between 0 and 1.

    Peaks at 1 when x equals the median, falls linearly to 0 at either bound,
    and is 0 outside them. Both branches are scaled independently, so the
    function handles an interval whose median is not centred -- which is the
    usual case for depths and slopes.

    Shared by the depth and the slope constraints.
    """
    if x < min_val or x > max_val:
        return 0
    # rescale x to [0, 1] on each side of the median, independently
    if x <= median:
        # left branch: from min_val to median
        t = (x - min_val) / (median - min_val)   # 0 at min_val, 1 at median
    else:
        # right branch: from median to max_val
        t = (max_val - x) / (max_val - median)   # 1 at median, 0 at max_val
    return float(t)

def bestCompromise(G, rim, node_id, depth_min, depth_max, depth_median, slope_median, slope_min, slope_max,
                    rimScore, upstreamScore, downstreamScore, z_invert, aberrant_set, z_diff=10):
    """Reconcile several constraints pulling the prediction in opposite ways.

    When more than one constraint is active, no single target value satisfies
    them all, so a direct search is run instead: the constraint scores are
    evaluated on 50 candidate values spread over `[z_invert - z_diff,
    z_invert + z_diff]` and the value maximising their **product** is kept.

    The product -- rather than a sum -- means a zero on any single constraint
    nullifies the whole combination, consistent with the idea that a violated
    physical constraint should invalidate the candidate outright.

    Returns None when at least two constraints are already unusable (missing or
    violated), in which case the caller leaves the node uncorrected. The graph
    traversal does not depend on the candidate value and is therefore run once,
    outside the loop.
    """
    noneValue = 0
    if not rimScore or rimScore == 0:
        noneValue += 1
    if not upstreamScore or upstreamScore == 0:
        noneValue += 1
    if not downstreamScore or downstreamScore == 0:
        noneValue += 1
    if noneValue >= 2:
        return None

    z_min = z_invert - z_diff
    z_max = z_invert + z_diff
    z_values = np.linspace(z_min, z_max, num=50)

    if rimScore and rimScore > 0:
        rim_scores = np.array([restrictionRimInvert(rim, z, depth_min, depth_max, depth_median) for z in z_values])
    else:
        rim_scores = np.ones(len(z_values))

    if (upstreamScore and upstreamScore > 0) or (downstreamScore and downstreamScore > 0):
        # The graph traversal does not depend on the candidate z -- run it once.
        (_, _, upstream_invert_mean, upstream_length_mean,
         downstream_invert_mean, downstream_length_mean) = restrictionInitTerm(
            G, node_id, z_invert, aberrant_set, slope_median, slope_min, slope_max
        )
        up_scores, down_scores = [], []
        for z in z_values:
            su, sd = slope_scores_from_means(
                z, upstream_invert_mean, upstream_length_mean,
                downstream_invert_mean, downstream_length_mean,
                slope_median, slope_min, slope_max,
            )
            up_scores.append(su if (upstreamScore and upstreamScore > 0) else 1)
            down_scores.append(sd if (downstreamScore and downstreamScore > 0) else 1)
        upstream_scores = np.array(up_scores)
        downstream_scores = np.array(down_scores)
    else:
        upstream_scores = np.ones(len(z_values))
        downstream_scores = np.ones(len(z_values))

    combined_scores = rim_scores * upstream_scores * downstream_scores
    best_idx = np.argmax(combined_scores)
    return z_values[best_idx]

def nearest_neighbors_upstream(G, node_id, accumulated_length = 0, nbNoeudVus = 0, results = None, max_depth=20, init_node = True):
    """Walk upstream until nodes with a known invert elevation are reached.

    Recursively follows every predecessor branch, accumulating the pipe lengths
    along the way, and stops a branch as soon as it reaches a node holding a
    value. Because the graph branches, several results may be returned -- one
    per branch.

    Each result is a tuple ``(node_id, invertElevation, accumulated_length,
    hops)``. The search is capped at `max_depth` hops, both to bound the cost
    and because a reference too far upstream carries little information about
    the target node.
    """
    if results == None:
        results = []
    if pd.notna(G.nodes[node_id].get("invertElevation")) and not init_node:
        return (node_id, G.nodes[node_id].get("invertElevation"), accumulated_length, nbNoeudVus)
        
    if len(list(G.predecessors(node_id))) == 0 or nbNoeudVus > max_depth:
        return

    predecessors = list(G.predecessors(node_id))
    for pred in predecessors:
        length = G[pred][node_id].get("length")
        current = nearest_neighbors_upstream(G, pred, accumulated_length = accumulated_length + length,
                                             nbNoeudVus = nbNoeudVus + 1, results = results, init_node = False)
        if type(current) == tuple:
            results.append(current)
    return results

def nearest_neighbors_downstream(G, node_id, accumulated_length = 0, nbNoeudVus = 0, results = None, max_depth=20, init_node = True):
    """Walk downstream until nodes with a known invert elevation are reached.

    Mirror image of `nearest_neighbors_upstream`, following successors instead
    of predecessors. Same return format and same depth cap.
    """
    if results == None:
        results = []
    if pd.notna(G.nodes[node_id].get("invertElevation")) and not init_node:
        return (node_id, G.nodes[node_id].get("invertElevation"), accumulated_length, nbNoeudVus)
        
    if len(list(G.successors(node_id))) == 0 or nbNoeudVus > max_depth:
        return

    predecessors = list(G.successors(node_id))
    for pred in predecessors:
        length = G[node_id][pred].get("length")
        current = nearest_neighbors_downstream(G, pred, accumulated_length = accumulated_length + length,
                                               nbNoeudVus = nbNoeudVus + 1, results = results, init_node = False)
        if type(current) == tuple:
            results.append(current)
    return results

def prediction_topologie(G, node_id, results_upstream, results_downstream,
                         depth_min = 0, depth_max = 0, depth_median = 0, alpha = 0, restriRim=False):
    """Predict one invert elevation from the topology alone (Method 2).

    Selects the nearest upstream and downstream reference (smallest cumulative
    distance along the pipes) among the branches returned by the traversal, and
    interpolates between them with an inverse-distance weighting.

    When only one side is available, the value is linearly extrapolated using
    the median network slope. When neither is, no prediction is possible and
    None is returned.

    Unlike Method 1, this method uses no spatial neighbourhood at all, which is
    precisely why it works where the missing values are spatially concentrated
    and spatial interpolation has nothing to interpolate from. Its weakness is
    the mirror image: it relies on two reference nodes only, making it less
    robust for *detecting* anomalies than an average over many neighbours.

    The depth constraint may optionally be applied to the result, exactly as in
    Method 1.
    """
    if results_upstream:
        result_upstream = results_upstream[0]
        for i in range(len(results_upstream)):
            if result_upstream[2] > results_upstream[i][2]:
                result_upstream = results_upstream[i]
                
    if results_downstream:
        result_downstream = results_downstream[0]
        for i in range(len(results_downstream)):
            if result_downstream[2] > results_downstream[i][2]:
                result_downstream = results_downstream[i]

    if results_upstream and results_downstream:
        invert = (
            ((result_upstream[1]/result_upstream[2]) + (result_downstream[1]/result_downstream[2])) /
            ((1 / result_upstream[2]) + (1 / result_downstream[2]))
        )
    elif results_upstream and not results_downstream:
        invert = result_upstream[1] - depth_median * result_upstream[2]
    elif not results_upstream and results_downstream:             
        invert = result_downstream[1] + depth_median * result_downstream[2]
    elif not results_upstream and not results_downstream:
        return

    if restriRim:
        rimScore = restrictionRimInvertTopo(G, node_id, invert,
                                               depth_min = depth_min, depth_max = depth_max, depth_median = depth_median)
        if rimScore is not None:
            if rimScore == 0:
                return
            target_invert = G.nodes[node_id].get("rimElevation") - depth_median
            invert = invert + alpha * (1 - rimScore) * (target_invert - invert)
        
    return invert

def restrictionRimInvertTopo(G, node_id, invert, depth_min = 0, depth_max = 0, depth_median = 0):
    """Depth-constraint score, reading the rim elevation from the graph.

    Identical to `restrictionRimInvert`, but takes the rim elevation from the
    node attributes of `G` instead of an array, since Method 2 operates
    directly on the graph rather than on a DataFrame.
    """
    if pd.isna(G.nodes[node_id].get("rimElevation")):
        return (None)
    depth = G.nodes[node_id].get("rimElevation") - invert
    if depth < depth_min or depth > depth_max:
        score = 0
    else:
        score = parabolic_score(depth, depth_median, depth_min, depth_max)
    return score

def hide_invert(G, pourcent = 0.1, seed = 42):
    """Mask a fraction of the known invert elevations, keeping the truth aside.

    The masking counterpart of `injectAnomalie`, used to validate the
    topological prediction: since no external ground truth exists for invert
    elevation, known values are hidden and the reconstruction is scored against
    them.

    Returns the masked graph and the list of ``(node_id, original attributes)``
    pairs. Only nodes actually holding a value are masked, so the requested
    percentage is reached in terms of *known* values.
    """
    random.seed(seed)
    G_test = G.copy()
    n = len(G_test.nodes())
    node_list = list(G_test.nodes())
    n_hide = 0
    verif = []
    list_random = list(i for i in range(n))
    while n_hide < int(n * pourcent):
        index = random.choice(list_random)
        list_random.remove(index)
        
        if pd.notna(G_test.nodes[node_list[index]].get("invertElevation")):
            verif.append((node_list[index],G_test.nodes[node_list[index]].copy()))
            G_test.nodes[node_list[index]]["invertElevation"] = None
            n_hide += 1
                
    return (G_test, verif)

def invert_prediction(G, depth_min = 0, depth_max = 0, depth_median = 0, alpha = 0, restriRim = False):
    """Apply the topological prediction (Method 2) to every missing node.

    Iterates over the whole graph, leaving nodes that already hold a value
    untouched and predicting the missing ones. Because the graph is modified as
    the iteration proceeds, a node predicted early can serve as a topological
    reference for the nodes that depend on it, letting the reconstruction
    propagate outwards from the well-documented zones.

    Returns a new graph; the input is left unchanged.
    """
    G_test = G.copy()
    for n in G_test:
        if pd.isna(G_test.nodes[n].get("invertElevation")):
            upstream = nearest_neighbors_upstream(G_test, n)
            downstream = nearest_neighbors_downstream(G_test, n)
            if upstream or downstream:
                G_test.nodes[n]["invertElevation"] = prediction_topologie(G_test, n, upstream, downstream, depth_min, depth_max,
                                                                          depth_median, alpha, restriRim)
    return G_test

def invert_result(G, verif, verbose = False):
    """Score a topological prediction run against the masked ground truth.

    Returns three indicators:

    * `err_abs` -- mean absolute error over the masked nodes that were
      successfully predicted;
    * `pctNbCorr` -- percentage of the *masked* nodes recovered. This stays
      close to 100% for almost any configuration, since masking only a small
      fraction leaves each masked node with plenty of known references, so it
      discriminates poorly between models;
    * `pct_nodes_still_nan` -- percentage of the *whole* network still lacking
      a value. This is the genuinely discriminating coverage measure, and the
      one that trades off against `err_abs` when ranking configurations.

    `err_abs` and `pctNbCorr` are None when no masked node could be scored,
    which happens when this is a real prediction run rather than a masked
    validation run.
    """
    err_abs = 0
    nb_cor = 0
    nan_after = 0
    for n in verif:
        if G.nodes[n[0]].get("invertElevation"):
            err_abs += abs(n[1].get("invertElevation") - G.nodes[n[0]].get("invertElevation"))
            nb_cor += 1
    err_abs = err_abs / nb_cor if nb_cor != 0 else None
    pctNbCorr = (nb_cor/len(verif)) * 100 if len(verif) != 0 else None

    for n in G.nodes():
        if pd.isna(G.nodes[n].get("invertElevation")):
            nan_after += 1
    pct_nodes_still_nan = (nan_after / len(G.nodes())) * 100
    
    if verbose:        
        print("Mean absolute error : ",err_abs)
        print("Percentage of masked nodes recovered : ", pctNbCorr)
        print("Percentage of network nodes still missing : ", pct_nodes_still_nan)
    return (err_abs, pctNbCorr, pct_nodes_still_nan)


def slopeCalcul(G):
    """Compute the pipe slopes of the network, for hydraulic diagnostics.

    For every node holding an invert elevation, finds its downstream
    neighbours and derives the slope from the elevation difference over the
    distance along the pipes. Each pipe is counted once (`seen_edges`), which
    matters because a double count would distort the distribution -- and the
    resulting statistics are used to check that the reconstruction has not
    introduced hydraulically implausible slopes.

    Pipes shorter than 5 m are excluded: over such a short distance, a small
    elevation error produces an enormous apparent slope.
    """
    slopes = []
    seen_edges = set()
    for node_id in G.nodes:
        if pd.notna(G.nodes[node_id].get("invertElevation")):
            downstream = nearest_neighbors_downstream(G, node_id)
            if downstream:
                for down in downstream:
                    edge = (node_id, down[0])  # down[0] is the neighbour's node_id
                    if edge not in seen_edges and (edge[1], edge[0]) not in seen_edges:
                        if pd.notna(down[1]) and (down[2] > 5):
                            slope = (G.nodes[node_id].get("invertElevation") - down[1]) / down[2]
                            slopes.append(slope)
                            seen_edges.add(edge)
    return slopes


def modelWithInjectedErrors(df_elev, radius, node_min, method, weight, seuil_clean = 5, thresholds = [40, 25, 15, 10, 5],
                            inter_threshold = [], err_min = 0, err_max = 150, pourcent = 0.1,
                            feature = "rimElevation", G = None, restriRim = False, restriInvert = False,
                            depth_min = 0, depth_max = 0, depth_median = 0, alpha = 0,
                            slope_min = 0, slope_max = 0, slope_median = 0, pred_NaN = False, verbose = True 
                           ):
    """Run one full injected-anomaly experiment, end to end.

    Chains the whole protocol for a single hyperparameter configuration: build
    a clean reference dataset, inject anomalies into it, run the
    detection/correction passes over the decreasing threshold sequence,
    optionally predict the missing values, and score the result.

    `thresholds` implements the coarse-to-fine strategy: large deviations are
    caught first, then progressively smaller ones. `inter_threshold` selects
    the thresholds at which a full evaluation is recorded, allowing several
    stopping points to be compared within a single run rather than re-running
    the pipeline for each. When it selects none, the run is evaluated once at
    the final threshold.

    The spatial neighbourhoods are computed once here and passed down, since
    they are identical for every pass.

    Returns a list of result dicts, one per evaluated threshold.
    """
    df_clean = clean_df_elev(df_elev, radius, seuil_clean, node_min, method, weight_method = weight, verbose = verbose)
    X, y = injectAnomalie(df_clean, err_min = err_min, err_max = err_max, pourcent = pourcent, feature = feature)

    neighbors_idx, neighbors_dist = precompute_neighbors(X, radius=radius)  # computed once, reused by every pass

    pred_rows=[]
    result = []
    for threshold in thresholds:
        X, pred_rows = detect_and_correct(X, radius, threshold, node_min, method, G, weight, pred_rows, feature, restriRim, restriInvert,
                                           depth_min, depth_max, depth_median, alpha, slope_min, slope_max, slope_median, neighbors_idx, neighbors_dist, verbose)
        if threshold in inter_threshold:
            X_NaN, pred_rows_NaN = X.copy(), pred_rows.copy()
            if pred_NaN:
                X_NaN, pred_rows_NaN = predictNaN(X, radius, node_min, method, weight, pred_rows, feature, G,
                                                restriRim, restriInvert, depth_min, depth_max, depth_median, alpha, slope_min, slope_max, slope_median, neighbors_idx, neighbors_dist)
            summary = evaluateReconstruction(X_NaN, y, pred_rows_NaN, radius, threshold, node_min, method, df_clean, weight, err_min, err_max, pourcent, feature,
                            restriRim, restriInvert, depth_min, depth_max, depth_median, alpha, slope_min, slope_max, slope_median,  verbose)
            result.append(summary)
    if len(result) == 0:
        X_NaN, pred_rows_NaN = X.copy(), pred_rows.copy()
        if pred_NaN:
            X_NaN, pred_rows_NaN = predictNaN(X, radius, node_min, method, weight, pred_rows, feature, G,
                                            restriRim, restriInvert, depth_min, depth_max, depth_median, alpha, slope_min, slope_max, slope_median, neighbors_idx, neighbors_dist)
        summary = evaluateReconstruction(X_NaN, y, pred_rows_NaN, radius, thresholds[-1], node_min, method, df_clean, weight, err_min, err_max, pourcent, feature,
                            restriRim, restriInvert, depth_min, depth_max, depth_median, alpha, slope_min, slope_max, slope_median,  verbose)
        result.append(summary)
    return result

def modelWithExternalReference(df_elev, radius, node_min, method, weight, thresholds = [40, 25, 15, 10, 5], inter_threshold = [], z_ref = None, feature = "rimElevation", epsg = "EPSG:2154", verbose = True):
    """Run one full real-data experiment against an external reference.

    Counterpart of `modelWithInjectedErrors` for the real-data setting: no
    anomaly is injected, the pipeline is applied to the data as recorded, and
    the reconstruction is compared with an independent elevation reference
    (`z_ref`).

    The threshold sequence typically stops higher than in the injected-anomaly
    setting, because real data contain few genuine errors and a very low
    threshold would flag mostly false positives.

    Returns a list of summary dicts, one per evaluated threshold.
    """
    X = df_elev.copy()

    neighbors_idx, neighbors_dist = precompute_neighbors(X, radius=radius)  # computed once, reused by every pass

    pred_rows=[]
    result = []
    for threshold in thresholds:
        X, pred_rows = detect_and_correct(X, radius, threshold, node_min, method, None, weight, pred_rows, neighbors_idx = neighbors_idx, neighbors_dist = neighbors_dist, verbose = verbose)
        if threshold in inter_threshold:
            X_NaN, pred_rows_NaN = predictNaN(X, radius, node_min, method, weight, pred_rows, neighbors_idx=neighbors_idx, neighbors_dist=neighbors_dist)
            gdf_after, summary = evaluate_real_case(X_NaN, z_ref, radius, node_min, method, weight, threshold, feature, epsg, verbose)
            result.append(summary)
    if len(result) == 0:
        X_NaN, pred_rows_NaN = predictNaN(X, radius, node_min, method, weight, pred_rows, neighbors_idx=neighbors_idx, neighbors_dist=neighbors_dist)
        gdf_after, summary = evaluate_real_case(X_NaN, z_ref, radius, node_min, method, weight, thresholds[-1], feature, epsg, verbose)
        result.append(summary)
    return result

