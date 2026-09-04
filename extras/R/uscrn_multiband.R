library(terra)
library(fs)
library(stringr)

root_bands <- paste0(
  "D:/SM-DeepLearning/datasets/Central valley/datasets/",
  "USCRN_validations/1km_samples"
)

dem_dir  <- file.path(root_bands, "DEM")
lulc_dir <- file.path(root_bands, "NLCD")
soil_dir <- file.path(root_bands, "soil")
lst_dir  <- file.path(root_bands, "LST")
smap_dir <- file.path(root_bands, "1km-r")
matched_9km_dir <- file.path(root_bands, "9km")

dir_create(out_dir)
dir_create(matched_9km_dir)

# -------------------------------------------------------------------
# Load static bands for one polygon ID.
# Return NULL if any required static raster is missing.
# -------------------------------------------------------------------
get_static_stack <- function(id) {
  
  files <- c(
    DEM  = file.path(dem_dir,  paste0("CV_SRTM90_Elevation_", id, ".tif")),
    LULC = file.path(lulc_dir, paste0("Annual_NLCD_LndCov_2024_", id, ".tif")),
    STC  = file.path(soil_dir, paste0("STC_M_1km_", id, ".tif"))
  )
  
  missing_files <- files[!file_exists(files)]
  
  if (length(missing_files) > 0) {
    return(NULL)
  }
  
  r <- rast(files)
  names(r) <- names(files)
  
  return(r)
}

# -------------------------------------------------------------------
# Load dated LST for one date and polygon ID.
# Return NULL instead of stopping if missing.
# -------------------------------------------------------------------
get_lst_layer <- function(date, id) {
  
  lst_file <- file.path(
    lst_dir,
    paste0("MYD11A1_LST_", date, "_", id, ".tif")
  )
  
  if (!file.exists(lst_file)) {
    return(NULL)
  }
  
  r_lst <- rast(lst_file)
  r_lst[r_lst == -9999] <- NaN
  names(r_lst) <- "LST_AM"
  
  return(r_lst)
}

# -------------------------------------------------------------------
# Read only correctly named resampled-SMAP files.
# Expected: SMAP-E_9km_AM_20240101_112.tif
# -------------------------------------------------------------------
smap_files <- list.files(
  smap_dir,
  pattern = "^SMAP-E_9km_AM_[0-9]{8}_.+\\.tif$",
  full.names = TRUE
)

total_files <- length(smap_files)

if (total_files == 0) {
  stop("No matching SMAP files found in: ", smap_dir)
}

message("Found ", total_files, " SMAP files.")

# Record skipped date-ID pairs for review.
skip_log <- data.frame(
  smap_file = character(),
  date = character(),
  id = character(),
  reason = character(),
  stringsAsFactors = FALSE
)

saved <- 0L
skipped <- 0L

# -------------------------------------------------------------------
# Build a multiband stack only for valid SMAP-LST date-ID pairs.
# -------------------------------------------------------------------
for (i in seq_along(smap_files)) {
  
  smap_file <- smap_files[i]
  filename <- basename(smap_file)
  
  message("Processing (", i, "/", total_files, "): ", filename)
  
  # Capture the date and complete ID from the SMAP filename.
  parsed <- str_match(
    filename,
    "^SMAP-E_9km_AM_([0-9]{8})_(.+)\\.tif$"
  )
  
  date <- parsed[1, 2]
  id <- parsed[1, 3]
  
  if (is.na(date) || is.na(id)) {
    skip_log <- rbind(
      skip_log,
      data.frame(filename, NA, NA, "Could not parse SMAP filename")
    )
    skipped <- skipped + 1L
    next
  }
  
  # Load matching LST for exactly the same date and ID.
  r_lst <- get_lst_layer(date, id)
  
  if (is.null(r_lst)) {
    message("  Skipped: matching LST is missing for date=", date, ", id=", id)
    
    skip_log <- rbind(
      skip_log,
      data.frame(filename, date, id, "Matching LST file missing")
    )
    
    skipped <- skipped + 1L
    next
  }
  
  # Load required static layers.
  r_static <- get_static_stack(id)
  
  if (is.null(r_static)) {
    message("  Skipped: one or more static layers are missing for id=", id)
    
    skip_log <- rbind(
      skip_log,
      data.frame(filename, date, id, "One or more static layers missing")
    )
    
    skipped <- skipped + 1L
    next
  }
  
  # Load the corresponding 1-km-resampled SMAP raster.
  r_smap <- rast(smap_file)
  names(r_smap) <- "SMAP_AM"
  
  # Every band must have identical CRS, extent, resolution, origin,
  # number of rows, and number of columns.
  smap_lst_ok <- compareGeom(
    r_smap, r_lst,
    stopOnError = FALSE
  )
  
  smap_static_ok <- compareGeom(
    r_smap, r_static,
    stopOnError = FALSE
  )
  
  if (!smap_lst_ok || !smap_static_ok) {
    message("  Skipped: raster geometries do not match for date=", date, ", id=", id)
    
    skip_log <- rbind(
      skip_log,
      data.frame(filename, date, id, "SMAP, LST, and/or static geometry mismatch")
    )
    
    skipped <- skipped + 1L
    next
  }
  
  # Band order:
  # 1 = resampled SMAP
  # 2 = dated LST
  # 3 = DEM
  # 4 = LULC
  # 5 = STC
  r_stack <- c(r_smap, r_lst, r_static)
  
  # Stack output: rename only the resolution label from 9km to 1km.
  out_name <- str_replace(filename, "_9km_", "_1km_")
  out_file <- file.path(out_dir, out_name)
  
  # Keep the matched SMAP raster under its original filename.
  copied_smap_file <- file.path(matched_9km_dir, filename)
  
  writeRaster(
    r_stack,
    out_file,
    overwrite = TRUE,
    wopt = list(
      gdal = c(
        "COMPRESS=LZW",
        "TILED=YES"
      )
    )
  )
  
  # Copy only SMAP files that produced a valid multiband stack.
  file_copy(
    smap_file,
    copied_smap_file,
    overwrite = TRUE
  )
  
  saved <- saved + 1L
  message("  Saved: ", out_file)
}

# -------------------------------------------------------------------
# Save skipped date-ID combinations for diagnosing missing inputs.
# -------------------------------------------------------------------
# skip_file <- file.path(out_dir, "skipped_smap_lst_pairs.csv")
# write.csv(skip_log, skip_file, row.names = FALSE)

cat("\nCompleted.\n")
cat("Saved stacks: ", saved, "\n")
cat("Skipped files: ", skipped, "\n")
# cat("Skip log: ", skip_file, "\n")
