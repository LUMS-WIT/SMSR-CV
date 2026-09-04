library(terra)
library(fs)
library(stringr)

terraOptions(progress = 1, memfrac = 0.6)

# -------------------------------------------------------------------
# Input paths
# -------------------------------------------------------------------
polys_path <- paste0(
  "D:/SM-DeepLearning/datasets/Central valley/datasets/",
  "USCRN_validations/ismn_smap_6x6_blocks/",
  "ismn_smap_6x6_blocks.shp"
)

stc_path <- paste0(
  "D:/SEBAL/datasets/soil/HiHydroSoil_250m/Top_Subsoil/STC/",
  "STC_M_250m_TOPSOIL.tif"
)

# One 1-km DEM raster per polygon/block.
# Example supplied by you:
# CV_SRTM90_Elevation_0.tif
ref_dir <- paste0(
  "D:/SM-DeepLearning/datasets/Central valley/datasets/",
  "USCRN_validations/1km_samples/DEM"
)

out_dir <- paste0(
  "D:/SM-DeepLearning/datasets/Central valley/datasets/",
  "USCRN_validations/1km_samples/soil"
)

dir_create(out_dir)

# -------------------------------------------------------------------
# Read data
# -------------------------------------------------------------------
polys <- vect(polys_path)
stc <- rast(stc_path)

if (is.na(crs(polys))) stop("Polygon shapefile has no CRS.")
if (is.na(crs(stc))) stop("STC raster has no CRS.")
if (!("id" %in% names(polys))) stop("Polygon shapefile needs an 'id' column.")

cat("Polygon CRS:\n", crs(polys), "\n\n")
cat("STC CRS:\n", crs(stc), "\n\n")
cat("STC resolution:", res(stc), "\n")

# Project only the vector to the STC CRS. Do not project the large
# categorical STC raster before cropping.
polys_stc <- if (same.crs(polys, stc)) {
  polys
} else {
  project(polys, crs(stc))
}

# -------------------------------------------------------------------
# Helper: bounding-box overlap check
# -------------------------------------------------------------------
overlaps_raster <- function(v, r) {
  ve <- ext(v)
  re <- ext(r)
  
  xmin(ve) < xmax(re) &&
    xmax(ve) > xmin(re) &&
    ymin(ve) < ymax(re) &&
    ymax(ve) > ymin(re)
}

# -------------------------------------------------------------------
# Extract and resample each block
# -------------------------------------------------------------------
saved <- 0L
skipped <- 0L

for (i in seq_len(nrow(polys))) {
  
  poly_id <- as.character(polys$id[i])
  safe_id <- str_replace_all(poly_id, "[^A-Za-z0-9_-]", "_")
  
  # Match the polygon ID to its existing 1-km DEM template.
  ref_file <- file.path(
    ref_dir,
    paste0("CV_SRTM90_Elevation_", safe_id, ".tif")
  )
  
  if (!file_exists(ref_file)) {
    message("Skipping ID ", poly_id, ": DEM reference file not found.")
    skipped <- skipped + 1L
    next
  }
  
  scale_ref <- rast(ref_file)
  
  if (is.na(crs(scale_ref))) {
    message("Skipping ID ", poly_id, ": DEM reference has no CRS.")
    skipped <- skipped + 1L
    next
  }
  
  # Confirm that the supplied reference is actually 1 km.
  if (any(abs(res(scale_ref) - 1000) > 1e-6)) {
    warning(
      "ID ", poly_id, ": reference resolution is ",
      paste(res(scale_ref), collapse = " x "),
      ", not 1000 m."
    )
  }
  
  # Use the vector in the STC CRS for cropping.
  block_stc <- polys_stc[i, ]
  
  if (!overlaps_raster(block_stc, stc)) {
    message("Skipping ID ", poly_id, ": block does not overlap STC raster.")
    skipped <- skipped + 1L
    next
  }
  
  # Crop first: this avoids reprojecting the full 250-m STC raster.
  stc_crop <- crop(stc, block_stc, snap = "out")
  stc_crop <- mask(stc_crop, block_stc)
  
  if (ncell(stc_crop) == 0) {
    message("Skipping ID ", poly_id, ": no STC cells after crop.")
    skipped <- skipped + 1L
    next
  }
  
  # Transfer the categorical STC values to the exact DEM target grid.
  #
  # "modal" = majority category when CRS is already the same.
  # "mode"  = majority category during a CRS transformation.
  if (same.crs(stc_crop, scale_ref)) {
    stc_1km <- resample(
      stc_crop,
      scale_ref,
      method = "modal",
      threads = TRUE
    )
  } else {
    stc_1km <- project(
      stc_crop,
      scale_ref,
      method = "mode",
      threads = TRUE
    )
  }
  
  # Reapply the precise polygon mask in the target/reference CRS.
  block_ref <- if (same.crs(polys[i, ], scale_ref)) {
    polys[i, ]
  } else {
    project(polys[i, ], crs(scale_ref))
  }
  
  stc_1km <- mask(stc_1km, block_ref)
  
  # Critical quality check: output must match the 1-km DEM grid exactly.
  if (!compareGeom(stc_1km, scale_ref, stopOnError = FALSE)) {
    stop(
      "ID ", poly_id,
      ": generated STC raster does not match its 1-km DEM reference grid."
    )
  }
  
  out_file <- file.path(
    out_dir,
    paste0("STC_M_1km_", safe_id, ".tif")
  )
  
  writeRaster(
    stc_1km,
    out_file,
    overwrite = TRUE,
    wopt = list(
      datatype = "INT2S",
      gdal = c("COMPRESS=DEFLATE")
    )
  )
  
  saved <- saved + 1L
  message("Saved: ", out_file)
}

cat("\nCompleted.\n")
cat("Saved: ", saved, "\n")
cat("Skipped: ", skipped, "\n")