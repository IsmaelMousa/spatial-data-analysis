import rasterio
import numpy                   as np
import pandas                  as pd
import geopandas               as gpd
import matplotlib.pyplot       as plt
from   rasterio.transform      import xy
from   sklearn.linear_model    import LinearRegression
from   sklearn.model_selection import KFold
from   sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from   math                    import sqrt, pi
from   collections             import deque


dem_path  = "data/arelev1.tif"
snow_path = "data/snowpoint.shp"

snow      = gpd.read_file(snow_path)


with rasterio.open(fp=dem_path) as src:
    dem           = src.read(1, masked=True).astype("float32")
    profile       = src.profile.copy()
    transform     = src.transform
    crs           = src.crs
    nodata        = src.nodata
    height, width = dem.shape
    resx, resy    = src.res


with rasterio.open(fp=dem_path) as src:
    coords = [(g.x, g.y) for g in snow.geometry]

    elev   = np.array([v[0] for v in src.sample(coords)], dtype="float32")


snow["ELEV"] = elev
snow         = snow[snow["ELEV"] != nodata]

xy_pts       = np.vstack([snow.geometry.x.values, snow.geometry.y.values]).T.astype("float32")
z_pts        = snow["SNOWDEPTH"].values.astype("float32")
e_pts        = snow["ELEV"].values.astype("float32")

lr           = LinearRegression().fit(e_pts.reshape(-1, 1), z_pts)
residuals    = z_pts - lr.predict(e_pts.reshape(-1, 1))


def idw(train_xy, train_z, pred_xy, power=2, eps=1e-6):
    d = np.sqrt(((pred_xy[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=2))

    w = 1.0 / np.maximum(d, eps) ** power

    return (w * train_z[None, :]).sum(axis=1) / w.sum(axis=1)


snow_pred       = np.full((height, width), np.nan, dtype="float32")

valid           = ~dem.mask

rows, cols      = np.where(valid)

regression_part = lr.predict(dem.data[valid].reshape(-1, 1))

block           = 5000
pred_res        = np.zeros(len(rows), dtype="float32")


for i in range(0, len(rows), block):
    j             = min(i + block, len(rows))

    xs, ys        = xy(transform, rows[i:j], cols[i:j], offset="center")

    pred_xy       = np.vstack([xs, ys]).T.astype("float32")

    pred_res[i:j] = idw(xy_pts, residuals, pred_xy)


snow_pred[valid] = np.clip(regression_part + pred_res, 0, None)

profile.update(dtype="float32", nodata=np.nan, compress="lzw")

with rasterio.open("data/snow_depth_pred_reg_idw.tif", "w", **profile) as dst: dst.write(snow_pred, 1)

def cv_idw(xy, z, power=2):
    """
    TODO
    :param xy:
    :param z:
    :param power:
    :return:
    """
    kf    = KFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(z)

    for tr, te in kf.split(xy): preds[te] = idw(xy[tr], z[tr], xy[te], power)

    return sqrt(mean_squared_error(z, preds)), mean_absolute_error(z, preds), r2_score(z, preds)


def cv_reg_idw(xy, z, elev, power=2):
    """
    TODO
    :param xy:
    :param z:
    :param elev:
    :param power:
    :return:
    """
    kf    = KFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(z)

    for tr, te in kf.split(xy):
        lr        = LinearRegression().fit(elev[tr].reshape(-1, 1), z[tr])

        res       = z[tr] - lr.predict(elev[tr].reshape(-1, 1))

        preds[te] = lr.predict(elev[te].reshape(-1, 1)) + idw(xy[tr], res, xy[te], power)

    return sqrt(mean_squared_error(z, preds)), mean_absolute_error(z, preds), r2_score(z, preds),


data             = [("IDW", *cv_idw(xy_pts, z_pts)), ("Regression + IDW", *cv_reg_idw(xy_pts, z_pts, e_pts))]
columns          = ["Method", "RMSE", "MAE", "R2"]

val_results      = pd.DataFrame(data=data, columns=columns)

dem_filled       = dem.filled(np.nan)
dz_drow, dz_dcol = np.gradient(dem_filled, resy, resx)

dz_dx            = dz_dcol
dz_dy            = -dz_drow

slope_rad        = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
slope_deg        = slope_rad * 180 / pi

aspect_rad       = np.arctan2(dz_dx, dz_dy)
aspect_deg       = (aspect_rad * 180 / pi + 360) % 360


def hillshade(slope, aspect, az=180, alt=30):
    """
    TODO
    :param slope:
    :param aspect:
    :param az:
    :param alt:
    :return:
    """
    az  = np.deg2rad(az)
    alt = np.deg2rad(alt)

    return np.clip(np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - np.deg2rad(aspect)), 0, 1)


shade_score = 1 - hillshade(slope_rad, aspect_deg)
northness   = (np.cos(np.deg2rad(aspect_deg)) + 1) / 2


def triangular(x, a, b, c):
    """
    TODO
    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    """
    y = np.zeros_like(x)
    y = np.where((x >= a) & (x < b), (x - a) / (b - a), y)
    y = np.where((x >= b) & (x <= c), (c - x) / (c - b), y)

    return np.clip(y, 0, 1)


slope_score             = triangular(slope_deg, 5, 25, 45)

valid                   = np.isfinite(snow_pred)
snow_norm               = np.zeros_like(snow_pred)
p1, p99                 = np.nanpercentile(snow_pred[valid], [1, 99])
snow_norm[valid]        = np.clip((snow_pred[valid] - p1) / (p99 - p1), 0, 1)

elev_mask               = dem_filled >= 2000
suitability             = np.full_like(snow_pred, np.nan)
suitability[valid]      = (0.5 * snow_norm[valid] + 0.25 * slope_score[valid] + 0.15 * northness[valid] + 0.1 * shade_score[valid])
suitability[~elev_mask] = np.nan


with rasterio.open("data/ski_resort_suitability.tif", "w", **profile) as dst: dst.write(suitability, 1)


threshold               = np.nanpercentile(suitability, 99.5)
mask                    = (suitability >= threshold) & np.isfinite(suitability)

labels                  = np.zeros_like(mask, dtype=int)
label                   = 0

for r in range(height):
    for c in range(width):
        if mask[r, c] and labels[r, c] == 0:
            label        += 1

            q            = deque([(r, c)])

            labels[r, c] = label

            while q:
                rr, cc = q.popleft()

                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    r2, c2 = rr + dr, cc + dc

                    if 0 <= r2 < height and 0 <= c2 < width:
                        if mask[r2, c2] and labels[r2, c2] == 0:
                            labels[r2, c2] = label
                            q.append((r2, c2))

clusters = []

for lab in range(1, label + 1):
    idx = np.where(labels == lab)

    if len(idx[0]) < 50: continue

    rmean = int(idx[0].mean())
    cmean = int(idx[1].mean())

    x, y  = xy(transform, rmean, cmean, offset="center")

    clusters.append({"cluster"  : lab                        ,
                     "cells"    : len(idx[0])                ,
                     "easting"  : x                          ,
                     "northing" : y                          ,
                     "elevation": dem_filled[rmean, cmean]   ,
                     "slope"    : slope_deg[rmean, cmean]    ,
                     "aspect"   : aspect_deg[rmean, cmean]   ,
                     "snow"     : snow_pred[rmean, cmean]    ,
                     "score"    : np.nanmax(suitability[idx]),})


clusters_df = pd.DataFrame(clusters).sort_values("score", ascending=False).head(5)

clusters_df.to_csv("top5_ski_candidates.csv", index=False)

plt.imshow(snow_pred)
plt.colorbar()
plt.savefig("snow_prediction.png", dpi=200)
plt.close()

plt.imshow(suitability)
plt.colorbar()
plt.savefig("ski_suitability.png", dpi=200)
plt.close()
