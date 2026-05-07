library(terra)
library(stringr)
library(fs)  # for file path handling

# === Paths ===
root_dir <- "D:/SM-DeepLearning/datasets/Central valley"

# Set input and output root folders
in_root <- file.path(root_dir, "SMAP-P-E-9km-AM")  # Input (coarser resolution)
out_root <- file.path(root_dir, "SMAP-P-E-1km-AM-r")  # Output (finer resolution)

scale_factor <- 9  # e.g., scaling factor is 9 for 9km → 1km
fine_res <- "1km"  # Update for output resolution

# List all .tif files recursively
tif_files <- list.files(in_root, pattern = "\\.tif$", recursive = TRUE, full.names = TRUE)

print(length(tif_files))

for (file in tif_files) {
  message("Processing: ", file)
  
  # Load coarse-resolution raster
  r_coarse <- rast(file)
  
  # Get resolution, extent, and CRS of coarse raster
  res_coarse <- res(r_coarse)
  ext_coarse <- ext(r_coarse)
  crs_coarse <- crs(r_coarse)
  
  # Compute finer resolution
  res_fine <- res_coarse / scale_factor
  
  # Compute aligned extent to ensure clean block alignment
  ncol_fine <- floor((ext_coarse[2] - ext_coarse[1]) / res_fine[1])
  nrow_fine <- floor((ext_coarse[4] - ext_coarse[3]) / res_fine[2])
  
  xmin <- ext_coarse[1]
  xmax <- ext_coarse[1] + ncol_fine * res_fine[1]
  ymin <- ext_coarse[3]
  ymax <- ext_coarse[3] + nrow_fine * res_fine[2]
  
  aligned_extent <- ext(xmin, xmax, ymin, ymax)
  
  # Build fine-resolution template
  r_fine_template <- rast(
    ext = aligned_extent,
    resolution = res_fine,
    crs = crs_coarse
  )
  
  # Resample coarse → fine
  r_fine <- resample(r_coarse, r_fine_template, method = "bilinear")
  
  # Output path logic: change "coarse" to "fine"
  rel_path <- path_rel(file, start = in_root)
  rel_out_file <- str_replace(rel_path, "9km", fine_res)
  out_file <- file.path(out_root, rel_out_file)
  
  dir_create(dirname(out_file))
  writeRaster(r_fine, out_file, overwrite = TRUE)
  message("Saved: ", out_file)
}
