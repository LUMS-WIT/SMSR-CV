library(terra)
library(fs)
library(stringr)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
smap_9km_dir <- paste0(
  "D:/SM-DeepLearning/datasets/Central valley/datasets/",
  "USCRN_validations/9km"
)

dem_ref_dir <- paste0(
  "D:/SM-DeepLearning/datasets/Central valley/datasets/",
  "USCRN_validations/1km_samples/DEM"
)

out_dir <- paste0(
  "D:/SM-DeepLearning/datasets/Central valley/datasets/",
  "USCRN_validations/1km_samples/1km-r"
)

dir_create(out_dir)

# -------------------------------------------------------------------
# Find input files
# Expected input:
# SMAP-E_9km_AM_20240101_{id}.tif
# -------------------------------------------------------------------
# Find every TIFF in the 9-km folder first.
all_tifs <- dir_ls(
  smap_9km_dir,
  glob = "*.tif",
  type = "file",
  recurse = FALSE
)

# Apply the anchored regex only to the filename, not the entire path.
smap_files <- all_tifs[
  str_detect(
    path_file(all_tifs),
    "^SMAP-E_9km_AM_[0-9]{8}_.+\\.tif$"
  )
]

if (length(smap_files) == 0) {
  print(path_file(all_tifs))  # diagnostic: show discovered TIFF names
  stop("No matching 9-km SMAP files found in: ", smap_9km_dir)
}

print(length(smap_files))

# -------------------------------------------------------------------
# Process each 9-km file
# -------------------------------------------------------------------
saved <- 0L
skipped <- 0L

for (smap_file in smap_files) {
  
  filename <- path_file(smap_file)
  
  # Extract:
  # date = 20240101
  # id   = everything after the date and underscore
  match <- str_match(
    filename,
    "^SMAP-E_9km_AM_([0-9]{8})_(.+)\\.tif$"
  )
  
  if (is.na(match[1, 1])) {
    message("Skipping unparsable filename: ", filename)
    skipped <- skipped + 1L
    next
  }
  
  date_tag <- match[1, 2]
  poly_id <- match[1, 3]
  
  # Corresponding 1-km DEM block, e.g. CV_SRTM90_Elevation_0.tif
  dem_file <- file.path(
    dem_ref_dir,
    paste0("CV_SRTM90_Elevation_", poly_id, ".tif")
  )
  
  if (!file_exists(dem_file)) {
    message(
      "Skipping ", filename,
      ": matching DEM reference not found: ", path_file(dem_file)
    )
    skipped <- skipped + 1L
    next
  }
  
  out_file <- file.path(
    out_dir,
    paste0("SMAP-E_9km_AM_", date_tag, "_", poly_id, ".tif")
  )
  
  smap_9km <- rast(smap_file)
  dem_ref <- rast(dem_file)
  
  if (is.na(crs(smap_9km))) {
    stop("Input SMAP file has no CRS: ", smap_file)
  }
  
  if (is.na(crs(dem_ref))) {
    stop("DEM reference has no CRS: ", dem_file)
  }
  
  # Use the DEM raster as the exact target grid:
  # CRS, extent, origin, 1-km resolution, rows, and columns.
  if (same.crs(smap_9km, dem_ref)) {
    
    smap_1km <- resample(
      smap_9km,
      dem_ref,
      method = "bilinear",
      threads = TRUE
    )
    
  } else {
    
    smap_1km <- project(
      smap_9km,
      dem_ref,
      method = "bilinear",
      threads = TRUE
    )
  }
  
  # The output must exactly match the corresponding DEM grid.
  if (!compareGeom(smap_1km, dem_ref, stopOnError = FALSE)) {
    stop(
      "Resampled output does not match DEM geometry for ID: ",
      poly_id
    )
  }
  
  writeRaster(
    smap_1km,
    out_file,
    overwrite = TRUE,
    wopt = list(
      datatype = "FLT4S",
      gdal = c("COMPRESS=DEFLATE")
    )
  )
  
  saved <- saved + 1L
  message("Saved: ", out_file)
}

cat("\nCompleted.\n")
cat("Saved: ", saved, "\n")
cat("Skipped: ", skipped, "\n")