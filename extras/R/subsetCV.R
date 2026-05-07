# =========================
# Load required libraries
# =========================
library(terra)
library(stringr)

SMAP_1km <- FALSE

# Target CRS
crs_target <- "EPSG:4326"

# =========================
# Input directories
# =========================
if (SMAP_1km) {
  tif_dir <- "D:/SM-DeepLearning/datasets/Central valley/SMAP-P-E-1km"
} else {
  tif_dir <- "D:/SM-DeepLearning/datasets/SMAP/2020_25_SM_E"
}

tif_files <- list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE)
total_files <- length(tif_files)
print(total_files)

# =========================
# Shapefile
# =========================
CV <- vect("D:/SM-DeepLearning/datasets/Central valley/shapefile/CV_shapefile/CV_polygon.shp")
shapefile <- CV
plot(shapefile)

# =========================
# Output directory
# =========================
if (SMAP_1km) {
  out_dir <- "D:/SM-DeepLearning/datasets/Central valley/SMAP-P-E-1km-AM"
} else {
  out_dir <- "D:/SM-DeepLearning/datasets/Central valley/SMAP-E_9km_CV_daily/cropped_layers"
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)


# =========================
# Processing loop
# =========================
i <- 0
for (file in tif_files) {
  i <- i + 1
  
  # ---- Extract date ----
  date_str <- str_extract(basename(file), "\\d{8}")
  if (is.na(date_str)) {
    message("Skipping file (no date found): ", basename(file))
    next
  }
  
  message("[", i, "/", total_files, "] Processing: ", date_str)
  
  # ---- Load raster ----
  r <- rast(file)
  
  # ---- SMAP 1 km: extract AM band FIRST ----
  if (SMAP_1km) {
    r <- r[[1]]   # Band 1 = AM
  }
  
  # ---- Check for empty raster ----
  if (all(is.na(values(r)))) {
    message("  Skipping: all values NA before cropping")
    next
  }
  
  # ---- Reproject shapefile to raster CRS ----
  shapefile_proj <- project(shapefile, crs(r))
  
  # ---- Crop & mask ----
  r_cropped <- crop(r, shapefile_proj)
  r_masked  <- mask(r_cropped, shapefile_proj)
  
  # ---- Check after masking ----
  if (all(is.na(values(r_masked)))) {
    message("  Skipping: all values NA after masking")
    next
  }
  
  # ---- Reproject to WGS84 if needed ----
  if (crs(r_masked) != crs_target) {
    r_masked <- project(r_masked, crs_target)
  }
  
  # ---- Output filename ----
  if (SMAP_1km) {
    out_file <- file.path(out_dir, paste0("SMAP-E_1km_AM_", date_str, ".tif"))
  } else {
    out_file <- file.path(out_dir, paste0("SMAP-E_9km_AM_", date_str, ".tif"))
  }
  
  # ---- Save ----
  writeRaster(r_masked, out_file, overwrite = TRUE)
  message("  Saved: ", out_file)
  
  # ---- Cleanup ----
  rm(r, r_cropped, r_masked, shapefile_proj)
  gc()
}

# =========================
# Test output
# =========================
test_file <- list.files(out_dir, pattern = "\\.tif$", full.names = TRUE)[2]
test <- rast(test_file)
plot(test)
