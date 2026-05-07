# === Add AOI from shapefile to the previous script ===
library(terra)

# === Paths ===
root_dir <- "D:/SM-DeepLearning/datasets/Central valley"

# Folder with rasters to align

ref_1km_path <- file.path(root_dir,"SMAP-P-E-1km-AM/SMAP-E_1km_AM_20150403.tif")
aoi_path     <- file.path(root_dir,"shapefile/CV_shapefile/CV_polygon.shp")          # <-- your shapefile
in_dir_9km   <- file.path(root_dir,"SMAP-E_9km_CV_daily/cropped_layers")
out_dir      <- file.path(root_dir,"SMAP-P-E-9km-AM")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

tif_files_9km <- list.files(in_dir_9km, pattern = "\\.tif$", full.names = TRUE)
length(tif_files_9km)

ref1km <- rast(ref_1km_path)
plot(ref1km)

# Load AOI and put it in the same CRS as the 1-km reference
aoi <- vect(aoi_path)
if (!same.crs(aoi, ref1km)) aoi <- project(aoi, crs(ref1km))
plot(aoi)

# --- helper: snap any extent to the 1-km grid (or to 9-km multiples) ---
snap_extent_to_grid <- function(E, r, mult = 1, snap = "out") {
  rx <- res(r)[1] * mult; ry <- res(r)[2] * mult
  R  <- ext(r); x0 <- R$xmin; y0 <- R$ymin
  fx <- function(x, step, origin, fun) origin + fun((x - origin)/step) * step
  if (snap == "out") {
    xmin <- fx(E$xmin, rx, x0, floor); xmax <- fx(E$xmax, rx, x0, ceiling)
    ymin <- fx(E$ymin, ry, y0, floor); ymax <- fx(E$ymax, ry, y0, ceiling)
  } else {
    xmin <- fx(E$xmin, rx, x0, round); xmax <- fx(E$xmax, rx, x0, round)
    ymin <- fx(E$ymin, ry, y0, round); ymax <- fx(E$ymax, ry, y0, round)
  }
  ext(xmin, xmax, ymin, ymax)
}

# 1) Build a 9-km template aligned to the 1-km grid, trimmed to AOI
#    (extent snapped OUT to full 9-km cells so edges fall on 9x1-km multiples)
E_aoi      <- ext(aoi)
E_aoi_9km  <- snap_extent_to_grid(E_aoi, ref1km, mult = 9, snap = "out")
template9  <- rast(E_aoi_9km, crs = crs(ref1km), resolution = res(ref1km) * 9)

# (Optional) If you want to mask to the polygon later, keep 'aoi' around.

# 2) Function to align a single 9-km raster to the template
align_one <- function(infile, tmpl, outdir, aoi_polygon = NULL) {
  r <- rast(infile)
  
  aligned <- if (same.crs(r, tmpl)) {
    resample(r, tmpl, method = "bilinear")
  } else {
    project(r, tmpl, method = "bilinear")
  }
  
  if (!is.null(aoi_polygon)) {
    aligned <- crop(aligned, aoi_polygon) |> mask(aoi_polygon)
  }
  
  # ---- rename using date only ----
  date_str <- stringr::str_extract(basename(infile), "\\d{8}")
  if (is.na(date_str)) {
    stop("No date found in filename: ", basename(infile))
  }
  
  outfile <- file.path(
    outdir,
    paste0("SMAP-E_9km_AM_", date_str, ".tif")
  )
  
  writeRaster(
    aligned,
    outfile,
    overwrite = TRUE,
    wopt = list(gdal = c("COMPRESS=LZW", "TILED=YES"))
  )
  
  outfile
}


# 3) Batch
files9 <- list.files(in_dir_9km, pattern = "\\.tif$|\\.tiff$", full.names = TRUE)
terraOptions(progress = 1)
invisible(lapply(files9, function(f) {
  cat("Aligning:", f, "...\n")
  tryCatch({
    out <- align_one(f, template9, out_dir, aoi_polygon = aoi)  # set to NULL if you don't want masking
    cat("  -> wrote:", out, "\n")
  }, error = function(e) cat("  !! Skipped:", conditionMessage(e), "\n"))
}))

aligned_files <- list.files(out_dir, pattern = "\\.tif$", full.names = TRUE)
length(aligned_files)